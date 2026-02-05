import cv2
import numpy as np
import time
import sqlite3
from datetime import datetime
from ultralytics import YOLO
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion

# ==========================================
# KONFIGURASI METODOLOGI
# ==========================================
SSD_CHECK_INTERVAL = 1      # Frekuensi SSD
WBF_IOU_THRESHOLD = 0.5     # Ambang batas IoU WBF
YOLO_WEIGHT = 2             # Bobot kepercayaan YOLO
SSD_WEIGHT = 1              # Bobot kepercayaan SSD
YOLO_CONF = 0.25
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

def save_to_db(vtype, tid, source="CCTV_Main"):
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
# CLASS: VEHICLE DETECTOR (UPDATED)
# ==========================================
class VehicleDetector:
    def __init__(self, lines_config, yolo_path='WeightModel/best.pt', ssd_path='WeightModel/ssd_validator.onnx'):
        # 1. Load Models (Share logic ideally, but separate instances are safer for threading)
        print("⏳ Loading YOLO...")
        try:
            self.yolo = YOLO(yolo_path)
        except:
            self.yolo = YOLO('yolov8n.pt')

        print("⏳ Loading SSD...")
        # ... (Kode SSD loading sama seperti sebelumnya) ...
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(ssd_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
        except:
            self.session = None

        # 2. Setup Counting & Tracking
        self.counter = LineCounter()
        
        # --- KONFIGURASI GARIS DINAMIS ---
        # Kita loop config yang dikirim dari app.py
        for line_name, coords in lines_config.items():
            # coords format: ((x1, y1), (x2, y2))
            self.counter.add_line(coords[0], coords[1], line_name)
        
        self.prev_pos = {}
        # Count storage fleksibel
        self.counts = {"mobil": 0, "truk": 0, "motor": 0}
        self.frame_id = 0
        
        # Definisi Kelas
        self.CAR_CLASSES = [2, 'car', 'cars', 'mobil', 'Pickup truck'] 
        self.TRUCK_CLASSES = [5, 7, 'bus', 'truck', 'truk', 'Medium Truck']
        self.MOTOR_CLASSES = [3, 'motorcycle', 'motor']

    # ... (Fungsi clamp_box & detect SSD sama seperti sebelumnya) ...
    def clamp_box(self, box):
        return [max(0.0, min(1.0, x)) for x in box]
        
    def detect_ssd(self, frame):
        if self.session is None: return [], [], []
        
        h, w = frame.shape[:2]
        input_img = cv2.resize(frame, (300, 300)).astype(np.uint8)
        input_tensor = np.expand_dims(input_img, axis=0)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Parsing output SSD (MobileNet Format)
        boxes_out, scores_out, classes_out = None, None, None
        for out in outputs:
            shape = out.shape
            if len(shape) == 3 and shape[2] == 4: boxes_out = out[0]
            elif len(shape) == 2:
                if out.dtype == np.float32 and np.max(out) <= 1.01: scores_out = out[0]
                else: classes_out = out[0]

        boxes, scores, labels = [], [], []
        if boxes_out is not None and scores_out is not None:
            for i in range(len(scores_out)):
                if scores_out[i] >= SSD_CONF:
                    # SSD output biasanya ymin, xmin, ymax, xmax. Kita ubah ke format WBF [x1, y1, x2, y2]
                    ymin, xmin, ymax, xmax = boxes_out[i]
                    boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                    scores.append(float(scores_out[i]))
                    labels.append(int(classes_out[i]) if classes_out is not None else 0)
        return boxes, scores, labels

    def detect_objects(self, frame, allowed_classes=None):
        """
        allowed_classes: List string, misal ['motor'] untuk hanya mendeteksi motor.
        Jika None, deteksi semua.
        """
        self.frame_id += 1
        h, w = frame.shape[:2]
        
        # --- 1. PREDIKSI YOLO ---
        y_res = self.yolo.predict(frame, conf=0.25, verbose=False)[0]
        y_boxes, y_scores, y_labels = [], [], []
        
        if y_res.boxes:
            for b in y_res.boxes:
                box = b.xyxyn.cpu().numpy()[0].tolist()
                y_boxes.append(self.clamp_box(box))
                y_scores.append(float(b.conf))
                y_labels.append(int(b.cls))

        # --- 2. PREDIKSI SSD (Jika session ada) ---
        s_boxes, s_scores, s_labels = [], [], []
        # (Implementasi SSD sama seperti sebelumnya)
        
        # --- 3. ENSEMBLE (WBF) ---
        # (Implementasi WBF sama seperti sebelumnya)
        # Untuk simplifikasi coding disini, anggap fused_boxes sudah ada
        # Jika tidak pakai WBF, gunakan y_boxes langsung
        boxes_list = [y_boxes, s_boxes]
        scores_list = [y_scores, s_scores]
        labels_list = [y_labels, s_labels]
        weights = [YOLO_WEIGHT, SSD_WEIGHT]

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, 
            weights=weights, iou_thr=WBF_IOU_THRESHOLD, conf_type='avg'
        )
        
        # --- 4. TRACKING ---
        results = self.yolo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.25)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()

            for box, tid, cid in zip(boxes, ids, clss):
                tid = int(tid)
                x1, y1, x2, y2 = map(int, box)
                
                # Tentukan Jenis Kendaraan
                cls_name = self.yolo.names[int(cid)]
                vtype = 'unknown'
                
                if cls_name in self.CAR_CLASSES or int(cid) == 2: vtype = 'mobil'
                elif cls_name in self.TRUCK_CLASSES or int(cid) in [5,7]: vtype = 'truk'
                elif int(cid) == 3 or cls_name == 'motorcycle': vtype = 'motor'
                
                # --- FILTERING KHUSUS MOTOR ---
                if allowed_classes and vtype not in allowed_classes:
                    continue # Skip jika bukan kelas yang diminta
                
                if vtype == 'unknown': continue

                # Titik Tengah Bawah
                curr_pt = (int((x1 + x2) / 2), y2)

                # Cek Line Crossing
                if tid in self.prev_pos:
                    if self.counter.check(tid, self.prev_pos[tid], curr_pt):
                        self.counts[vtype] += 1
                        # save_to_db(vtype, tid, "CCTV_Motor") # Aktifkan DB jika perlu
                        cv2.circle(frame, curr_pt, 15, (0, 0, 255), -1)

                self.prev_pos[tid] = curr_pt

                # Gambar Box
                color = (255, 100, 0) if vtype == 'motor' else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{vtype}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame = self.counter.draw(frame)
        return frame, self.counts, 0