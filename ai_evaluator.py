import os
import cv2
import glob
import numpy as np
import time
import random  # <--- [BARU] Tambahkan library random

class ModelEvaluator:
    def __init__(self, detector_instance, images_path, labels_path):
        self.detector = detector_instance
        self.images_path = images_path
        self.labels_path = labels_path
        
        # Mapping Kelas (Sesuaikan dengan urutan ID YOLO 0-3)
        self.classes = ['Golongan II', 'Golongan IVA', 'Golongan IVB', 'Golongan VB'] 
        self.matrix_classes = self.classes + ['Background']
        
        # Folder Output Debug
        self.debug_dir = "static/debug_results"
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

    def get_ground_truth(self, filename, img_w, img_h):
        label_file = os.path.join(self.labels_path, filename.rsplit('.', 1)[0] + '.txt')
        boxes = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) >= 5:
                        cls_id = int(data[0])
                        cx, cy, w, h = map(float, data[1:5])
                        
                        x1 = int((cx - w/2) * img_w)
                        y1 = int((cy - h/2) * img_h)
                        x2 = int((cx + w/2) * img_w)
                        y2 = int((cy + h/2) * img_h)
                        
                        boxes.append({'class_id': cls_id, 'box': [x1, y1, x2, y2], 'matched': False})
        return boxes

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def save_debug_image(self, frame, gt_boxes, preds, img_name, model_type):
        """Fungsi menggambar kotak GT (Hijau) vs Prediksi (Merah/Biru)"""
        debug_img = frame.copy()
        
        # 1. Gambar Ground Truth (HIJAU)
        for gt in gt_boxes:
            x1, y1, x2, y2 = gt['box']
            cls_name = self.classes[gt['class_id']] if gt['class_id'] < len(self.classes) else str(gt['class_id'])
            
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"GT: {cls_name}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 2. Gambar Prediksi (Warna Warni sesuai Model)
        color = (0, 0, 255) # Default Merah
        if model_type == 'yolov8': color = (255, 0, 255) # Ungu
        elif model_type == 'ensemble': color = (0, 255, 255) # Kuning
        
        for pred in preds:
            x1, y1, x2, y2 = pred['box']
            cls_id = pred['class_id']
            score = pred['score']
            cls_name = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
            
            # Geser teks sedikit agar tidak menumpuk dengan GT
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_img, f"{cls_name} {score:.2f}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Simpan Gambar
        save_path = os.path.join(self.debug_dir, f"{model_type}_{img_name}")
        cv2.imwrite(save_path, debug_img)
        print(f"📸 Debug image saved: {save_path}")

    def run(self, model_type='ensemble', samples=20, conf_thresh=0.5):
        # Ambil semua file gambar dulu
        image_files = glob.glob(os.path.join(self.images_path, "*.jpg"))
        if not image_files:
            image_files = glob.glob(os.path.join(self.images_path, "*.png"))
            
        # --- [MODIFIKASI] LOGIKA RANDOM SAMPLING ---
        total_files = len(image_files)
        
        if total_files > samples:
            # Ambil sampel acak sebanyak 'samples'
            image_files = random.sample(image_files, samples)
            print(f"🎲 Randomly selected {samples} images from {total_files} total files.")
        else:
            # Jika jumlah file kurang dari samples, ambil semua
            print(f"⚠️ Warning: Requested {samples} samples, but only found {total_files}. Using all images.")
        # -------------------------------------------
        
        tp, fp, fn = 0, 0, 0
        total_time = 0
        
        num_classes = len(self.classes)
        bg_idx = num_classes
        confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

        # Counter untuk limit penyimpanan gambar debug
        saved_debug_count = 0 
        MAX_DEBUG_IMAGES = 3  # Simpan 3 gambar per model

        for img_path in image_files:
            img_name = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            h, w = frame.shape[:2]

            # 1. Get Ground Truth
            gt_boxes = self.get_ground_truth(img_name, w, h)
            
            # 2. Get Prediction
            start_t = time.time()
            preds = self.detector.predict_raw(frame, model_type=model_type)
            total_time += (time.time() - start_t)
            
            # Filter Confidence
            preds = [p for p in preds if p['score'] >= conf_thresh]
            
            # --- DEBUGGING VISUAL ---
            # Simpan gambar untuk 3 sample PERTAMA (dari hasil acak tadi)
            if saved_debug_count < MAX_DEBUG_IMAGES:
                self.save_debug_image(frame, gt_boxes, preds, img_name, model_type)
                saved_debug_count += 1
            # ------------------------

            # 3. Matching Logic
            for pred in preds:
                p_box = pred['box']
                p_cls = pred['class_id']
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt in enumerate(gt_boxes):
                    iou = self.compute_iou(p_box, gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= 0.5:
                    if best_gt_idx != -1:
                        gt_cls = gt_boxes[best_gt_idx]['class_id']
                        r = gt_cls if gt_cls < num_classes else bg_idx
                        c = p_cls if p_cls < num_classes else bg_idx
                        
                        if gt_cls == p_cls:
                            if not gt_boxes[best_gt_idx]['matched']:
                                tp += 1
                                gt_boxes[best_gt_idx]['matched'] = True
                                confusion_matrix[r][c] += 1
                            else:
                                fp += 1 
                                confusion_matrix[bg_idx][c] += 1
                        else:
                            fp += 1
                            confusion_matrix[r][c] += 1
                else:
                    fp += 1
                    if p_cls < num_classes:
                        confusion_matrix[bg_idx][p_cls] += 1

            for gt in gt_boxes:
                if not gt['matched']:
                    fn += 1
                    gt_cls = gt['class_id']
                    if gt_cls < num_classes:
                        confusion_matrix[gt_cls][bg_idx] += 1

        epsilon = 1e-6
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        accuracy = tp / (tp + fp + fn + epsilon) * 100 
        fps = len(image_files) / (total_time + epsilon)

        return {
            "accuracy": round(accuracy, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "fps": round(fps, 1),
            "map50": round((precision + recall) / 2 * 100, 2),
            "total_tp": tp,
            "total_fp": fp,
            "total_fn": fn,
            "confusionMatrix": confusion_matrix.tolist()
        }