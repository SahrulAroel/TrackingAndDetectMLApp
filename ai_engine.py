import cv2
import numpy as np
import time
import sqlite3
from datetime import datetime
from ultralytics import YOLO
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion

# ==========================================
# KONFIGURASI UMUM
# ==========================================
SSD_CHECK_INTERVAL = 1      # Cek SSD setiap 1 frame
WBF_IOU_THRESHOLD = 0.5
YOLO_WEIGHT = 2             # Bobot YOLO lebih besar
SSD_WEIGHT = 1
YOLO_CONF = 0.25            # Confidence threshold standar (digunakan di Deteksi & Evaluasi)
SSD_CONF = 0.30

# ==========================================
# DATABASE HELPER
# ==========================================
def init_db():
    conn = sqlite3.connect('laporan_kendaraan.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detail_kendaraan (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        waktu_melintas TEXT,
        jenis_kendaraan TEXT,
        track_id INTEGER,
        video_source TEXT
    )''')
    conn.commit()
    conn.close()
    print("✅ Database Terinisialisasi")

def save_to_db(vtype, tid, source="System"):
    try:
        conn = sqlite3.connect('laporan_kendaraan.db')
        conn.cursor().execute(
            'INSERT INTO detail_kendaraan (waktu_melintas, jenis_kendaraan, track_id, video_source) VALUES (?, ?, ?, ?)',
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), vtype, int(tid), str(source))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ DB Error: {e}")

# ==========================================
# GEOMETRY UTILS
# ==========================================
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def get_line_side(p, line_start, line_end):
    return (line_end[0] - line_start[0]) * (p[1] - line_start[1]) - \
           (line_end[1] - line_start[1]) * (p[0] - line_start[0])

# ==========================================
# CLASS: SSD DETECTOR
# ==========================================
class SSDDetector:
    def __init__(self, model_path):
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = ['detection_boxes', 'detection_scores', 'detection_classes']
        except Exception as e:
            print(f"⚠️ Gagal Load SSD ({model_path}): {e}")
            self.session = None

    def detect(self, frame):
        if self.session is None: return [], [], []
        
        target_size = (640, 640) 
        input_img = cv2.resize(frame, target_size)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.uint8)
        input_tensor = np.expand_dims(input_img, axis=0)

        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        except Exception as e:
            return [], [], []
        
        raw_boxes = outputs[0][0]
        raw_scores = outputs[1][0]
        raw_classes = outputs[2][0]

        boxes, scores, labels = [], [], []
        
        for i in range(len(raw_scores)):
            score = float(raw_scores[i])
            if score >= SSD_CONF:
                box = raw_boxes[i]
                ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
                x1, y1, x2, y2 = float(xmin), float(ymin), float(xmax), float(ymax)

                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))
                
                if (x2 - x1) > 0.01 and (y2 - y1) > 0.01:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    labels.append(int(raw_classes[i]))

        return boxes, scores, labels

# ==========================================
# CLASS: LINE COUNTER
# ==========================================
class LineCounter:
    def __init__(self):
        self.lines = {}
        self.counted_ids = set()

    def add_line(self, p1, p2, name):
        self.lines[name] = {'p1': p1, 'p2': p2, 'color': (0, 255, 255)}

    def check(self, tid, prev_pt, curr_pt):
        if tid in self.counted_ids: return False
        
        for name, ln in self.lines.items():
            if intersect(prev_pt, curr_pt, ln['p1'], ln['p2']):
                prev_side = get_line_side(prev_pt, ln['p1'], ln['p2'])
                curr_side = get_line_side(curr_pt, ln['p1'], ln['p2'])
                
                if prev_side > 0 and curr_side <= 0: 
                    self.counted_ids.add(tid)
                    ln['color'] = (0, 0, 255)
                    return True
        return False

    def draw(self, frame):
        for name, ln in self.lines.items():
            cv2.line(frame, ln['p1'], ln['p2'], ln['color'], 3)
            mid_x = int((ln['p1'][0] + ln['p2'][0]) / 2)
            mid_y = int((ln['p1'][1] + ln['p2'][1]) / 2)
            cv2.putText(frame, name, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return frame

# ==========================================
# MAIN CLASS: VEHICLE DETECTOR
# ==========================================
class VehicleDetector :
    def __init__(self, lines_config=None, yolo_path='WeightModel/WeightBaru/best.onnx', ssd_path='WeightModel/WeightBaru/model.onnx'):
        
        # 1. Load YOLO
        print("⏳ Loading YOLO...")
        self.yolo_path = yolo_path
        try:
            self.yolo = YOLO(yolo_path, task='detect')
        except:
            print("⚠️ Custom YOLO not found, downloading yolov8n...")
            self.yolo = YOLO('yolov8n.pt', task='detect')

        # 2. Load SSD
        print("⏳ Loading SSD...")
        self.ssd = SSDDetector(ssd_path)

        # 3. Setup Line Counter
        self.counter = LineCounter()
        if lines_config:
            for line_name, coords in lines_config.items():
                self.counter.add_line(coords[0], coords[1], line_name)
        
        # 4. Inisialisasi Variable
        self.prev_pos = {}
        self.counts = {
            "Golongan II": 0, "Golongan IVA": 0, "Golongan IVB": 0, "Golongan VB": 0
        }
        self.frame_id = 0
        
        # 5. Definisi ID Kelas
        self.MOTOR_CLASSES = [3, 'motorcycle', 'motor', 'Golongan II', 0] 
        self.CAR_CLASSES = [2, 'car', 'cars', 'mobil', 'Golongan IVA', 1] 
        self.PICKUPTRUCK_CLASSES = ['Pickup truck', 'pickup' , 'Golongan IVB', 2] 
        self.MEDIUMTRUCK_CLASSES = [5, 7, 'bus', 'truck', 'truk', 'Medium Truck', 'Golongan VB', 3]
        
        init_db()

    def clamp_box(self, box):
        return [max(0.0, min(1.0, x)) for x in box]

    # [NEW] FUNGSI RESET TRACKER
    # Panggil fungsi ini jika berganti dari Mode Evaluasi ke Mode Live Detection
    # Di dalam class VehicleDetector file ai_engine.py

    def reset_tracker(self):
        print("🔄 MEMORY WIPE: Resetting YOLO Tracker State...")
        self.prev_pos = {} # Hapus history garis
        self.counts = {    # Optional: Reset counter jika diperlukan per sesi
            "Golongan II": 0, "Golongan IVA": 0, "Golongan IVB": 0, "Golongan VB": 0
        }
    
        # KUNCI UTAMA: Reload Model untuk menghapus state Kalman Filter
        try:
            # Load ulang model ke memori (ini menghapus cache tracker lama)
            self.yolo = YOLO(self.yolo_path, task='detect')
        except Exception as e:
            print(f"⚠️ Gagal reset model: {e}")

    # --- FUNGSI 1: LIVE VIEW (OPTIMIZED) ---
    def detect_objects(self, frame, allowed_classes=None):
        self.frame_id += 1
        h, w = frame.shape[:2]
        
        # A. YOLO TRACKING (Jalan Sekali Saja)
        # Menggunakan persist=True untuk tracking kontinyu
        # Memastikan YOLO tidak melakukan resize otomatis yang aneh-aneh
        results = self.yolo.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640, conf=YOLO_CONF)
        
        # Ambil data untuk WBF (Ensemble)
        y_boxes, y_scores, y_labels = [], [], []
        if results[0].boxes:
            for b in results[0].boxes:
                # Koordinat normalisasi (0-1) untuk WBF
                y_boxes.append(self.clamp_box(b.xyxyn.cpu().numpy()[0].tolist()))
                y_scores.append(float(b.conf))
                y_labels.append(int(b.cls))

        # B. SSD DETECTOR
        s_boxes, s_scores, s_labels_raw = [], [], []
        if self.frame_id % SSD_CHECK_INTERVAL == 0:
            s_boxes, s_scores, s_labels_raw = self.ssd.detect(frame)
        
        s_labels = [label - 1 for label in s_labels_raw] # Normalisasi ID

        # C. WBF (ENSEMBLE) - Dilakukan untuk penggabungan presisi
        boxes_list = [y_boxes, s_boxes]
        scores_list = [y_scores, s_scores]
        labels_list = [y_labels, s_labels]
        weights = [YOLO_WEIGHT, SSD_WEIGHT]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, 
            weights=weights, iou_thr=WBF_IOU_THRESHOLD, conf_type='avg'
        )

        # D. VISUALISASI & COUNTING
        # Kita menggunakan ID dari hasil tracking YOLO
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()

            for box, tid, cid in zip(boxes, ids, clss):
                tid = int(tid)
                x1, y1, x2, y2 = map(int, box)
                
                cid_int = int(cid)
                vtype = 'unknown'
                
                if cid_int == 0: vtype = 'Golongan II'
                elif cid_int == 1: vtype = 'Golongan IVA'
                elif cid_int == 2: vtype = 'Golongan IVB'
                elif cid_int == 3: vtype = 'Golongan VB'
                
                if allowed_classes and vtype not in allowed_classes: continue 
                if vtype == 'unknown': continue

                curr_pt = (int((x1 + x2) / 2), y2)

                # Counting Logic
                if tid in self.prev_pos:
                    if self.counter.check(tid, self.prev_pos[tid], curr_pt):
                        if vtype in self.counts:
                            self.counts[vtype] += 1
                            save_to_db(vtype, tid) 
                            cv2.circle(frame, curr_pt, 15, (0, 0, 255), -1)

                self.prev_pos[tid] = curr_pt

                # Drawing
                color = (0, 255, 0)
                if vtype == 'Golongan II':    color = (255, 100, 0)
                elif vtype == 'Golongan IVA': color = (0, 255, 0)
                elif vtype == 'Golongan IVB': color = (0, 255, 255)
                elif vtype == 'Golongan VB':  color = (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{tid} {vtype}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame = self.counter.draw(frame)
        return frame, self.counts, 0

    # --- FUNGSI 2: UNTUK EVALUASI (RAW DATA) ---
    def predict_raw(self, frame, model_type='ensemble'):
        h, w = frame.shape[:2]

        # 1. YOLO (FIX: Gunakan YOLO_CONF 0.25, bukan 0.01)
        # Menggunakan predict (bukan track) karena ini evaluasi per frame static
        y_res = self.yolo.predict(frame, conf=YOLO_CONF, verbose=False)[0]
        
        y_boxes, y_scores, y_labels = [], [], []
        if y_res.boxes:
            for b in y_res.boxes:
                y_boxes.append(self.clamp_box(b.xyxyn.cpu().numpy()[0].tolist()))
                y_scores.append(float(b.conf))
                y_labels.append(int(b.cls))

        if model_type == 'yolov8':
            results = []
            for i in range(len(y_scores)):
                x1, y1, x2, y2 = y_boxes[i]
                results.append({
                    'box': [int(x1*w), int(y1*h), int(x2*w), int(y2*h)],
                    'score': y_scores[i],
                    'class_id': y_labels[i]
                })
            return results

        # 2. SSD
        s_boxes_raw, s_scores, s_labels_raw = self.ssd.detect(frame)
        s_labels = [label - 1 for label in s_labels_raw] 
        s_boxes = s_boxes_raw

        if model_type == 'ssd':
            results = []
            for i in range(len(s_scores)):
                x1, y1, x2, y2 = s_boxes[i]
                results.append({
                    'box': [int(x1*w), int(y1*h), int(x2*w), int(y2*h)],
                    'score': s_scores[i],
                    'class_id': s_labels[i]
                })
            return results

        # 3. ENSEMBLE
        boxes_list = [y_boxes, s_boxes]
        scores_list = [y_scores, s_scores]
        labels_list = [y_labels, s_labels]
        weights = [YOLO_WEIGHT, SSD_WEIGHT]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, 
            weights=weights, iou_thr=WBF_IOU_THRESHOLD, conf_type='avg'
        )

        results = []
        for i in range(len(fused_scores)):
            x1, y1, x2, y2 = fused_boxes[i]
            results.append({
                'box': [int(x1*w), int(y1*h), int(x2*w), int(y2*h)],
                'score': float(fused_scores[i]),
                'class_id': int(fused_labels[i])
            })
        
        return results