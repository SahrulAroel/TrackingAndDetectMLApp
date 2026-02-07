from flask import Flask, render_template, jsonify, Response, request
import cv2
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sqlite3
import glob
import base64
from flask_cors import CORS

# Import Engine dari ai_engine.py
from ai_engine import VehicleDetector, init_db
from ai_evaluator import ModelEvaluator

# ==========================================
# 1. KONFIGURASI APLIKASI & DATABASE
# ==========================================
app = Flask(__name__)
CORS(app)

# Inisialisasi Database
init_db()

# Path Video (Sesuaikan nama file kamu)
VIDEO_SOURCE_MOBIL = 'video/TEST1.mp4'
VIDEO_SOURCE_MOTOR = 'video/TestMotor.mp4' 

DB_PATH = 'laporan_kendaraan.db'

# Path dataset - sesuaikan dengan struktur folder Anda
BASE_DATASET_PATH = 'dataset'
TRAIN_IMG_PATH = os.path.join(BASE_DATASET_PATH, 'train', 'images')
TRAIN_LBL_PATH = os.path.join(BASE_DATASET_PATH, 'train', 'labels')
VALID_IMG_PATH = os.path.join(BASE_DATASET_PATH, 'valid', 'images')
VALID_LBL_PATH = os.path.join(BASE_DATASET_PATH, 'valid', 'labels')
TEST_IMG_PATH = os.path.join(BASE_DATASET_PATH, 'test', 'images')
TEST_LBL_PATH = os.path.join(BASE_DATASET_PATH, 'test', 'labels')

# Global variable untuk menyimpan history evaluasi
evaluation_history = []

# Kelas kendaraan sesuai dataset YOLO
VEHICLE_CLASSES = ['Golongan II', 'Golongan IVA', 'Golongan IVB', 'Golongan VB']

# ==========================================
# 2. KONFIGURASI DETEKTOR AI (DUAL INSTANCE)
# ==========================================

# A. Konfigurasi Detektor MOBIL
lines_mobil = {
    "Line_In": ((534, 720), (90, 635)),
    "Line_Out": ((1185, 338), (1229, 408))
}
print("🔧 Inisialisasi Detektor Mobil...")
detector_mobil = VehicleDetector(lines_config=lines_mobil)

# B. Konfigurasi Detektor MOTOR
lines_motor = {
    "garis_motor": ((1, 413), (953, 413))
}
print("🔧 Inisialisasi Detektor Motor...")
detector_motor = VehicleDetector(lines_config=lines_motor)

# ==========================================
# 3. GLOBAL VARIABLES (STATE)
# ==========================================
initial_counts = {
    "Golongan II": 0, 
    "Golongan IVA": 0, 
    "Golongan IVB": 0, 
    "Golongan VB": 0
}

# State Mobil
mobil_running = False
mobil_counts = initial_counts.copy()
mobil_fps = 0

# State Motor
motor_running = False
motor_counts = initial_counts.copy()
motor_fps = 0

# ==========================================
# 4. ROUTING HALAMAN (PAGE ROUTES)
# ==========================================

@app.route('/')
@app.route('/dashboard')
def dashboard():
    """Halaman Utama Dashboard"""
    return render_template('dashboard_home.html')

@app.route('/antrian-mobil')
def antrian_mobil():
    """Halaman Monitoring Mobil"""
    return render_template('antrian_mobil.html', now=datetime.now())

@app.route('/antrian-motor')
def antrian_motor():
    """Halaman Monitoring Motor"""
    return render_template('antrian_motor.html', now=datetime.now())

@app.route('/laporan')
def laporan():
    """Halaman Laporan Data"""
    return render_template('laporan.html')

@app.route('/evaluasi')
def evaluasi():
    """Halaman Evaluasi Model"""
    return render_template('evaluasi.html')

# ==========================================
# 5. VIDEO STREAMING ROUTES
# ==========================================

@app.route('/video_feed')
def video_feed():
    def generate():
        global mobil_counts, mobil_fps
        
        # --- [FIX 1] RESET SEBELUM MULAI STREAM ---
        # Ini mencegah "ingatan" dari halaman sebelumnya merusak akurasi awal
        detector_mobil.reset_tracker()
        
        src = VIDEO_SOURCE_MOBIL if os.path.exists(VIDEO_SOURCE_MOBIL) else 0
        cap = cv2.VideoCapture(src)
        
        prev_time = time.time()

        while True:
            success, frame = cap.read()
            
            # --- [FIX 2] RESET SAAT VIDEO LOOPING ---
            if not success:
                detector_mobil.reset_tracker() # Reset lagi saat video ulang dari awal
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()
            
            if success:
                frame = cv2.resize(frame, (1280, 720))
                
                if mobil_running:
                    # Deteksi berjalan dengan memori yang segar
                    frame, counts, _ = detector_mobil.detect_objects(frame)
                    mobil_counts = counts
                    
                    # Hitung FPS
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                    prev_time = curr_time
                    mobil_fps = fps
                    
                    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    mobil_fps = 0
                    cv2.putText(frame, "AI STOPPED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                break
                
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_motor')
def video_feed_motor():
    """Video Stream Khusus Halaman Motor"""
    def generate():
        global motor_counts, motor_fps
        
        # 1. RESET TRACKER (MOTOR)
        detector_motor.reset_tracker()
        
        src = VIDEO_SOURCE_MOTOR if os.path.exists(VIDEO_SOURCE_MOTOR) else VIDEO_SOURCE_MOBIL
        cap = cv2.VideoCapture(src)
        prev_time = time.time()
        
        while True:
            success, frame = cap.read()
            if not success:
                # 2. RESET TRACKER SAAT LOOP
                detector_motor.reset_tracker()
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()
            
            if success:
                frame = cv2.resize(frame, (1280, 720))

                if motor_running:
                    # LOGIKA KHUSUS: Hanya deteksi kelas 'motor'
                    frame, counts, _ = detector_motor.detect_objects(frame, allowed_classes=['Golongan II'])
                    motor_counts = counts
                    
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                    prev_time = curr_time
                    motor_fps = fps
                    
                    cv2.putText(frame, "MODE: MOTOR ONLY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
                else:
                    motor_fps = 0
                    cv2.putText(frame, "AI STOPPED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# 6. API ENDPOINTS (MOBIL)
# ==========================================

@app.route('/api/status')
def get_status():
    return jsonify({
        "running": mobil_running,
        "counts": mobil_counts,
        "fps": mobil_fps,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/start', methods=['POST'])
def start_mobil():
    global mobil_running
    mobil_running = True
    return jsonify({"status": "started"})

@app.route('/api/stop', methods=['POST'])
def stop_mobil():
    global mobil_running
    mobil_running = False
    return jsonify({"status": "stopped"})

# ==========================================
# 7. API ENDPOINTS (MOTOR)
# ==========================================

@app.route('/api/motor/status')
def get_motor_status():
    return jsonify({
        "running": motor_running,
        "counts": motor_counts,
        "fps": motor_fps
    })

@app.route('/api/motor/start', methods=['POST'])
def start_motor():
    global motor_running
    motor_running = True
    return jsonify({"status": "started"})

@app.route('/api/motor/stop', methods=['POST'])
def stop_motor():
    global motor_running
    motor_running = False
    return jsonify({"status": "stopped"})

# ==========================================
# 8. API ENDPOINTS (LAINNYA / SYSTEM)
# ==========================================

@app.route('/api/system_metrics')
def get_system_metrics():
    """Simulasi data resource server"""
    return jsonify({
        "cpu_usage": np.random.uniform(20, 80),
        "memory_usage": np.random.uniform(30, 70),
        "disk_usage": np.random.uniform(40, 90),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/laporan_data')
def get_laporan_data():
    """Mengambil data dari SQLite untuk Grafik Laporan"""
    try:
        conn = sqlite3.connect('laporan_kendaraan.db')
        cursor = conn.cursor()
        # Query contoh: ambil data per tanggal
        query = """
        SELECT date(waktu_melintas) as tanggal, jenis_kendaraan, COUNT(*) 
        FROM detail_kendaraan 
        GROUP BY tanggal, jenis_kendaraan
        ORDER BY tanggal DESC LIMIT 50
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        print(f"DB Error: {e}")
        return jsonify([])
    
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Agar bisa akses kolom pakai nama (dict-like)
    return conn

@app.route('/api/vehicle-reports')
def get_vehicle_reports():
    try:
        # 1. Ambil Parameter Filter dari URL
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        vehicle_type = request.args.get('vehicle_type')

        conn = get_db_connection()
        cursor = conn.cursor()

        # 2. Bangun Query SQL Dinamis
        query = "SELECT * FROM detail_kendaraan WHERE 1=1"
        params = []

        # Filter Tanggal (Support format YYYY-MM-DD HH:MM:SS di database)
        if start_date and end_date:
            query += " AND waktu_melintas BETWEEN ? AND ?"
            # Tambahkan jam awal dan akhir hari agar akurat
            params.extend([f"{start_date} 00:00:00", f"{end_date} 23:59:59"])
        
        # Filter Jenis Kendaraan
        if vehicle_type and vehicle_type != 'all':
            query += " AND jenis_kendaraan = ?"
            params.append(vehicle_type)

        query += " ORDER BY waktu_melintas DESC"

        # 3. Eksekusi Query
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # 4. Konversi Data ke Dictionary
        record_list = []
        for row in rows:
            record_list.append(dict(row))
            
        # 5. Siapkan Data Ringkasan (Opsional, tapi membantu frontend)
        # Kita kirim data mentah (records), biar JS yang hitung chart-nya agar sinkron.
        
        conn.close()
        
        return jsonify({
            'status': 'success',
            'records': record_list,
            'count': len(record_list)
        })

    except sqlite3.OperationalError:
        return jsonify({'status': 'error', 'message': 'Tabel database tidak ditemukan. Pastikan tabel "detail_kendaraan" sudah dibuat.'}), 500
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
def get_daily_trend_data(cursor, days=7):
    """Get data for daily trend chart"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days-1)
    
    query = """
        SELECT DATE(waktu_melintas) as date, COUNT(*) as count 
        FROM detail_kendaraan 
        WHERE DATE(waktu_melintas) BETWEEN ? AND ?
        GROUP BY DATE(waktu_melintas)
        ORDER BY date
    """
    
    cursor.execute(query, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
    results = cursor.fetchall()
    
    # Create labels and data arrays
    labels = []
    data = []
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        label = current_date.strftime('%d %b')
        
        count = 0
        for result in results:
            if result[0] == date_str:
                count = result[1]
                break
        
        labels.append(label)
        data.append(count)
        current_date += timedelta(days=1)
    
    return {'labels': labels, 'data': data}

def get_vehicle_distribution_data(cursor):
    """Get data for vehicle distribution chart"""
    cursor.execute("SELECT jenis_kendaraan, COUNT(*) FROM detail_kendaraan GROUP BY jenis_kendaraan")
    results = cursor.fetchall()
    
    labels = [row[0] for row in results]
    data = [row[1] for row in results]
    
    return {'labels': labels, 'data': data}

def get_hourly_distribution_data(cursor, date):
    """Get hourly distribution for a specific date"""
    query = """
        SELECT strftime('%H', waktu_melintas) as hour, COUNT(*) 
        FROM detail_kendaraan 
        WHERE DATE(waktu_melintas) = ?
        GROUP BY strftime('%H', waktu_melintas)
        ORDER BY hour
    """
    
    cursor.execute(query, (date,))
    results = cursor.fetchall()
    
    # Create full 24-hour array
    labels = [f"{str(i).zfill(2)}:00" for i in range(24)]
    data = [0] * 24
    
    for hour, count in results:
        hour_int = int(hour)
        data[hour_int] = count
    
    return {'labels': labels, 'data': data}

# ==========================================
# 7. ENDPOINT EVALUASI (REAL DATA)
# ==========================================
@app.route('/api/run_evaluation', methods=['POST'])
def run_evaluation():
    try:
        data = request.json
        samples = int(data.get('samples', 20))
        conf = float(data.get('confidence', 0.5))
        
        print(f"🚀 Memulai Evaluasi Real pada {samples} sampel...")
        
        # Inisialisasi Evaluator dengan detector_mobil (Global Object)
        evaluator = ModelEvaluator(detector_mobil, VALID_IMG_PATH, VALID_LBL_PATH)
        
        # 1. Evaluasi YOLOv8
        print(" -> Testing YOLOv8...")
        yolo_res = evaluator.run(model_type='yolov8', samples=samples, conf_thresh=conf)
        
        # 2. Evaluasi SSD
        print(" -> Testing SSD...")
        ssd_res = evaluator.run(model_type='ssd', samples=samples, conf_thresh=conf)
        
        # 3. Evaluasi Ensemble
        print(" -> Testing Ensemble...")
        ensemble_res = evaluator.run(model_type='ensemble', samples=samples, conf_thresh=conf)
        
        # ======================================================================
        # 🔥 POIN PENTING: RESET TRACKER DI SINI 🔥
        # ======================================================================
        # Karena proses evaluasi di atas memanggil .predict() berkali-kali, 
        # state tracker di dalam YOLO pasti kacau. Kita reset SEBELUM kembali ke UI.
        print("🔄 Auto-Resetting Tracker after Evaluation...")
        detector_mobil.reset_tracker() 
        # ======================================================================

        return jsonify({
            "status": "success",
            "sample_count": samples,
            "yolov8": yolo_res,
            "ssd": ssd_res,
            "ensemble": ensemble_res
        })

    except Exception as e:
        print(f"❌ Error Evaluasi: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
# ==========================================
# 9. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    # Pastikan folder ada
    for folder in ['templates', 'static', 'WeightModel', 'VideoTesting']:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            
    print("\n" + "="*30)
    print("SISTEM MONITORING LALU LINTAS AKTIF")
    print("Akses Dashboard: http://localhost:5000/")
    print("="*30 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)