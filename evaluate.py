import os
import time
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory
from ultralytics import YOLO
import tensorflow as tf
from ensemble_boxes import weighted_boxes_fusion
import json

app = Flask(__name__)

# ==========================================
# KONFIGURASI KELAS & MODEL
# ==========================================
CLASS_NAMES = ['Golongan II', 'Golongan IVA', 'Golongan IVB', 'Golongan VB']
NUM_CLASSES = len(CLASS_NAMES)

# Path dataset
VALID_IMAGES_PATH = 'Dataset/test/images'
VALID_LABELS_PATH = 'Dataset/test/labels'

# ==========================================
# SCIENTIFIC HELPER FUNCTIONS (REVISED)
# ==========================================
def compute_iou_pixels(box1, box2):
    """
    Compute IoU antara dua bounding boxes dalam PIXEL [x1, y1, x2, y2].
    WAJIB menggunakan pixel atau aspect-ratio corrected coordinates untuk Skripsi.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def load_ground_truth(img_file, img_w, img_h):
    label_path = os.path.join(
        VALID_LABELS_PATH,
        img_file.replace('.jpg', '.txt')
    )

    boxes = []
    classes = []

    if not os.path.exists(label_path):
        return boxes, classes

    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            cls = int(parts[0])
            coords = parts[1:]

            xs = coords[0::2]
            ys = coords[1::2]

            # normalized → pixel
            x_min = min(xs) * img_w
            y_min = min(ys) * img_h
            x_max = max(xs) * img_w
            y_max = max(ys) * img_h

            boxes.append([x_min, y_min, x_max, y_max])
            classes.append(cls)

    return boxes, classes



def calculate_ap_per_class(tp_list, conf_list, num_gt_per_class):
    """
    Menghitung Average Precision (AP) standar PASCAL VOC/COCO.
    tp_list: List of 1 (TP) or 0 (FP) sorted by confidence
    """
    if len(tp_list) == 0 or num_gt_per_class == 0:
        return 0.0

    # Accumulate TPs and FPs
    tp = np.cumsum(tp_list)
    fp = np.cumsum(1 - np.array(tp_list))
    
    # Calculate precision and recall
    recalls = tp / num_gt_per_class
    precisions = tp / (tp + fp + 1e-16) # epsilon safety
    
    # Add sentinel values for integration (Standard VOC2010+)
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    
    # Smooth precision curve (make it monotonically decreasing)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
        
    # Calculate Area Under Curve (AUC)
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap * 100

# ==========================================
# MODEL MANAGER (UPDATED)
# ==========================================
class ModelManager:
    def __init__(self):
        self.yolo_model = YOLO('WeightModel/WeightBaru/best.pt')
        self.ssd_model = tf.saved_model.load('WeightModel/WeightBaru/exported_models/my_model/saved_model')

    def predict_yolo(self, image):
        h, w = image.shape[:2]
        results = self.yolo_model(image, conf=0.5, iou=0.45, verbose=False)[0]

        if results.boxes is None:
            return [], [], []

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        return boxes.tolist(), scores.tolist(), classes.tolist()

    def predict_ssd(self, image):
        if self.ssd_model is None:
            return [], [], []

        h, w = image.shape[:2]
        
        # === PERBAIKAN DI SINI ===
        # Ubah BGR (OpenCV) ke RGB (TensorFlow)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_tensor = tf.convert_to_tensor([image_rgb], dtype=tf.uint8)
        detections = self.ssd_model(input_tensor)
        # =========================

        boxes = detections['detection_boxes'][0].numpy()      # [y1,x1,y2,x2]
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)

        # ==========================================
        # MAPPING YANG SUDAH DIPERBAIKI (FINAL)
        # ==========================================
        SSD_TO_YOLO_MAP = {
            1: 0,  # SSD Medium Truck (1) -> YOLO Medium Truck (0)
            2: 1,  # SSD Pickup truck (2) -> YOLO Pickup truck (1)
            3: 2,  # SSD Rickshaw (3)     -> YOLO Rickshaw (2)
            4: 3,  # SSD cars (4)         -> YOLO cars (3)   # SSD motorcycle (5)   -> YOLO motorcycle (4)
        }

        boxes_out, scores_out, classes_out = [], [], []

        for box, score, cls in zip(boxes, scores, classes):
            if score < 0.35:
                continue
            if cls not in SSD_TO_YOLO_MAP:
                continue

            y1, x1, y2, x2 = box
            boxes_out.append([
                x1 * w,
                y1 * h,
                x2 * w,
                y2 * h
            ])
            scores_out.append(float(score))
            classes_out.append(SSD_TO_YOLO_MAP[cls])

        return boxes_out, scores_out, classes_out


    def predict_ensemble_wbf(self, image):
        # WBF butuh NORMALIZED coordinates [0,1]
        h, w = image.shape[:2]
        
        # Get raw pixel predictions
        b_yolo, s_yolo, c_yolo = self.predict_yolo(image)
        b_ssd, s_ssd, c_ssd = self.predict_ssd(image)
        
        if not b_yolo and not b_ssd: return [], [], []

        # Normalize boxes for WBF
        norm_yolo = [[b[0]/w, b[1]/h, b[2]/w, b[3]/h] for b in b_yolo]
        norm_ssd = [[b[0]/w, b[1]/h, b[2]/w, b[3]/h] for b in b_ssd]

        try:
            boxes, scores, labels = weighted_boxes_fusion(
                [norm_yolo, norm_ssd],
                [s_yolo, s_ssd],
                [c_yolo, c_ssd],
                weights=[2, 1], # YOLO biasanya lebih akurat, beri bobot lebih
                iou_thr=0.6,
                skip_box_thr=0.01
            )
            # Convert back to Pixels
            pixel_boxes = [[b[0]*w, b[1]*h, b[2]*w, b[3]*h] for b in boxes]
            return pixel_boxes, scores.tolist(), labels.astype(int).tolist()
        except Exception:
            return b_yolo, s_yolo, c_yolo

model_manager = ModelManager()
# ==========================================
# EVALUATION LOGIC (FIXED)
# ==========================================
# ==========================================
# PASTE INI DI app.py (GANTI FUNCTION LAMA)
# ==========================================
def evaluate_model(model_name, image_files, sample_limit=50, iou_threshold=0.5):
    """
    REVISI: Evaluasi dengan matriks 6x6 (termasuk Background untuk FP/FN)
    """
    import random
    manager = model_manager 

    # 1. Shuffle Dataset
    image_files = image_files.copy()
    random.shuffle(image_files)
    image_files = image_files[:sample_limit]

    # 2. Setup Variables
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # Matriks 6x6: 
    # Index 0-4: Kelas Kendaraan
    # Index 5: Background (Kosong)
    bg_idx = NUM_CLASSES  
    confusion_matrix = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)

    detections = []
    ground_truths = []
    inference_times = []
    processed_images = 0

    # 3. Loop Images
    for image_id, img_file in enumerate(image_files):
        img_path = os.path.join(VALID_IMAGES_PATH, img_file)
        if not os.path.exists(img_path): continue

        image = cv2.imread(img_path)
        if image is None: continue

        processed_images += 1
        h, w = image.shape[:2]

        # -- Inference --
        infer_start = time.time()
        if model_name == 'yolov8':
            pred_boxes, pred_scores, pred_classes = manager.predict_yolo(image)
        elif model_name == 'ssd':
            pred_boxes, pred_scores, pred_classes = manager.predict_ssd(image)
        else:
            pred_boxes, pred_scores, pred_classes = manager.predict_ensemble_wbf(image)
        infer_end = time.time()
        inference_times.append(infer_end - infer_start)

        # -- Load GT --
        gt_boxes, gt_classes = load_ground_truth(img_file, w, h)

        # Simpan GT untuk mAP calculation nanti
        for gt_box, gt_cls in zip(gt_boxes, gt_classes):
            ground_truths.append({
                'image_id': image_id, 'cls': gt_cls, 'box': gt_box
            })

        gt_matched = [False] * len(gt_boxes)

        # -- MATCHING LOOP (Pred vs GT) --
        for p_box, p_score, p_cls in zip(pred_boxes, pred_scores, pred_classes):
            detections.append({
                'image_id': image_id, 'cls': p_cls, 'conf': p_score, 'box': p_box
            })

            best_iou = 0.0
            best_gt_idx = -1

            # Cari GT terbaik untuk prediksi ini
            for j, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_matched[j]: continue # GT sudah dipakai
                # Opsional: Jika ingin strict class matching, uncomment baris bawah
                # if gt_cls != p_cls: continue 

                iou = compute_iou_pixels(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            # Penentuan TP atau FP
            if best_iou >= iou_threshold and best_gt_idx != -1:
                # TRUE POSITIVE (atau Salah Klasifikasi antar objek)
                total_tp += 1
                gt_matched[best_gt_idx] = True
                
                gt_actual_cls = gt_classes[best_gt_idx]
                # Isi matrix: Baris Actual, Kolom Predicted
                confusion_matrix[gt_actual_cls][p_cls] += 1
            else:
                # FALSE POSITIVE (Deteksi Hantu)
                # Actual: Background (5), Predicted: Kelas Mobil (0-4)
                total_fp += 1
                confusion_matrix[bg_idx][p_cls] += 1

        # -- CEK FN (GT yang tidak terdeteksi) --
        for k, matched in enumerate(gt_matched):
            if not matched:
                # FALSE NEGATIVE (Missed Detection)
                # Actual: Mobil (0-4), Predicted: Background (5)
                total_fn += 1
                gt_actual_cls = gt_classes[k]
                confusion_matrix[gt_actual_cls][bg_idx] += 1

    # 4. Calculate Metrics
    avg_inference_time = np.mean(inference_times) * 1000 if inference_times else 0
    fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    # mAP Calculation (Standard)
    ap_per_class = []
    for cls in range(NUM_CLASSES):
        cls_dets = [d for d in detections if d['cls'] == cls]
        cls_gts = [g for g in ground_truths if g['cls'] == cls]
        if not cls_gts: continue
        
        cls_dets.sort(key=lambda x: -x['conf'])
        tp = np.zeros(len(cls_dets))
        fp = np.zeros(len(cls_dets))
        used = set()

        for i, det in enumerate(cls_dets):
            candidates = [g for g in cls_gts if g['image_id'] == det['image_id']]
            best_iou, best_idx = 0, -1
            for idx, gt in enumerate(candidates):
                if (det['image_id'], idx) in used: continue
                iou = compute_iou_pixels(det['box'], gt['box'])
                if iou > best_iou: best_iou, best_idx = iou, idx
            
            if best_iou >= iou_threshold and best_idx != -1:
                tp[i] = 1
                used.add((det['image_id'], best_idx))
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / len(cls_gts)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
        ap = np.trapz(precisions, recalls)
        ap_per_class.append(ap)

    map50 = np.mean(ap_per_class) * 100 if ap_per_class else 0

    # 5. Return Result dengan Label 'Background'
    # Tambahkan label 'Background' untuk frontend
    extended_labels = CLASS_NAMES + ['Background']

    return {
        'accuracy': round((total_tp / (total_tp + total_fp + total_fn) * 100), 1) if (total_tp + total_fp + total_fn) else 0,
        'precision': round(precision * 100, 1),
        'recall': round(recall * 100, 1),
        'f1_score': round(f1_score * 100, 1),
        'map50': round(map50, 1),
        'inference_time': round(avg_inference_time, 1),
        'fps': round(fps, 1),
        'confusion_matrix': confusion_matrix.tolist(),
        'labels': extended_labels, # Label sudah 6 item
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }
# ==========================================
# FLASK ROUTES (UNCHANGED)
# ==========================================
@app.route('/')
@app.route('/evaluasi')
def index():
    return render_template('evaluasi.html')

@app.route('/api/run_evaluation', methods=['POST'])
def run_evaluation():
    try:
        data = request.get_json()
        sample_count = min(data.get('samples', 20), 273)  # Maks 273 sesuai dataset
        
        image_files = [f for f in os.listdir(VALID_IMAGES_PATH) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            return jsonify({'status': 'error', 'message': 'Dataset not found'}), 400
        
        results = {}
        # Gunakan sample yang sama untuk semua model agar fair
        eval_images = image_files[:sample_count]
        
        for model in ['yolov8', 'ssd', 'ensemble']:
            print(f"Evaluating {model}...")
            results[model] = evaluate_model(model, eval_images, sample_count)
        
        return jsonify({
            'status': 'success',
            'yolov8': results['yolov8'],
            'ssd': results['ssd'],
            'ensemble': results['ensemble'],
            'sample_count': len(eval_images)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500
        
@app.route('/debug_ssd')
def debug_ssd():
    import os
    
    # Ambil 1 gambar sample
    image_files = os.listdir(VALID_IMAGES_PATH)
    if not image_files: return "No images"
    
    # Pilih gambar yang ramai (cari manual nama filenya atau random)
    img_name = image_files[0] 
    img_path = os.path.join(VALID_IMAGES_PATH, img_name)
    
    # 1. Baca Gambar
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    
    # 2. Prediksi SSD (Pastikan pakai fix RGB yang saya sarankan sebelumnya)
    # Copy paste logika predict_ssd disini atau panggil methodnya
    # INGAT: Konversi BGR ke RGB dulu untuk input model!
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor([image_rgb], dtype=tf.uint8)
    detections = model_manager.ssd_model(input_tensor)
    
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)
    
    # 3. Gambar Kotak di Image Asli (BGR)
    for box, score, cls in zip(boxes, scores, classes):
        if score < 0.25: continue # Threshold
        
        y1, x1, y2, x2 = box
        # Konversi ke pixel
        x_min, y_min = int(x1 * w), int(y1 * h)
        x_max, y_max = int(x2 * w), int(y2 * h)
        
        # Gambar kotak
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
        # Tulis Label Asli SSD (Tanpa Mapping dulu)
        label_text = f"ID:{cls} Conf:{score:.2f}"
        cv2.putText(image, label_text, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Simpan hasil
    output_path = 'static/debug_ssd.jpg'
    cv2.imwrite(output_path, image)
    
    return f"Cek gambar di folder static/debug_ssd.jpg. <br> Gambar: {img_name}"

@app.route('/api/evaluation/export', methods=['POST'])
def export_results():
    try:
        data = request.get_json()
        export_data = {
            'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models': data.get('models'),
            'dataset_info': {'path': VALID_IMAGES_PATH}
        }
        filename = f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('exports', exist_ok=True)
        filepath = os.path.join('exports', filename)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        return jsonify({'status': 'success', 'download_url': f'/exports/{filename}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/exports/<filename>')
def download_export(filename):
    return send_from_directory('exports', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)