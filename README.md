# 🚗 Vehicle Counting & Classification System (AI-Powered)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green?style=for-the-badge&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-white?style=for-the-badge&logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-yellow?style=for-the-badge)

Sistem pemantauan lalu lintas cerdas berbasis web yang dibangun menggunakan **Flask** dan **Deep Learning**. Aplikasi ini mampu mendeteksi, menghitung, dan mengklasifikasikan kendaraan secara *real-time* menggunakan model **YOLOv8**, **SSD**, dan metode **Ensemble**.

Sistem ini dirancang khusus untuk memonitor antrian kendaraan (Mobil & Motor) dan menyediakan dashboard analitik serta pelaporan data otomatis.

## ✨ Fitur Utama

* **🕵️ Real-time Detection:** Deteksi kendaraan akurat menggunakan YOLOv8 dan SSD MobileNet.
* **🔢 Automatic Counting:** Penghitung otomatis berdasarkan *Region of Interest* (Line Counter) untuk jalur masuk dan keluar.
* **📊 Web Dashboard:** Antarmuka monitoring interaktif untuk streaming video CCTV/File.
* **🏍️ Dual Mode Monitoring:** Mendukung monitoring terpisah untuk **Antrian Mobil** dan **Antrian Motor**.
* **📈 Smart Reporting:** Visualisasi data tren harian, distribusi jenis kendaraan, dan jam sibuk menggunakan Chart.js.
* **📑 Evaluation Module:** Fitur bawaan untuk menguji performa model (mAP & FPS) pada dataset validasi.
* **💾 Database Integration:** Penyimpanan riwayat deteksi otomatis ke SQLite (`laporan_kendaraan.db`).

---

## 🛠️ Tech Stack

* **Backend:** Flask, Python
* **Computer Vision:** OpenCV (`cv2`), NumPy
* **AI Core:** PyTorch, Ultralytics YOLOv8, TensorFlow (untuk SSD jika digunakan)
* **Database:** SQLite3
* **Frontend:** HTML5, Bootstrap, Chart.js (diasumsikan untuk visualisasi)

---

## 🚀 Persyaratan Sistem & Instalasi

### 1. Prasyarat
Pastikan Anda telah menginstal:
* Python 3.8 - 3.10
* [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (Wajib untuk akselerasi GPU)
* Git

### 2. Clone Repository
```bash
git clone [https://github.com/username-anda/nama-repo-anda.git](https://github.com/username-anda/nama-repo-anda.git)
cd nama-repo-anda
```

### 3. Setup Environment (PENTING: NVIDIA CUDA) ⚠️
Agar model AI berjalan cepat menggunakan GPU NVIDIA, Anda HARUS menginstal PyTorch versi CUDA. Jangan instal PyTorch biasa.

#### Langkah 1: Buat Virtual Environment (Disarankan)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

#### Langkah 2: Instal PyTorch dengan dukungan CUDACek versi CUDA Anda (misal 11.8 atau 12.1) dan jalankan perintah yang sesuai:
```
Contoh untuk CUDA 11.8
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```
Jika menggunakan versi CUDA lain, cek command yang sesuai di pytorch.org.

#### Langkah 3: Instal Dependensi Lainnya
```
pip install flask opencv-python ultralytics numpy matplotlib flask-cors pandas
```

## ⚙️ Konfigurasi Project
Sebelum menjalankan aplikasi, sesuaikan konfigurasi path video dan dataset di file app.py:
app.py
```
# 1. Sesuaikan path video sumber (Bisa diganti URL RTSP CCTV)
VIDEO_SOURCE_MOBIL = 'video/assa.mp4'
VIDEO_SOURCE_MOTOR = 'video/TestMotor.mp4' 

# 2. Pastikan struktur folder dataset sesuai untuk fitur Evaluasi
BASE_DATASET_PATH = 'dataset'
```

Pastikan struktur folder dataset Anda terlihat seperti ini:
```
dataset/
├── train/
├── valid/
│   ├── images/
│   └── labels/
└── test/
```
## ▶️ Cara Menjalankan

Inisialisasi Database
Aplikasi akan otomatis membuat file laporan_kendaraan.db saat pertama kali dijalankan berkat fungsi init_db().
Jalankan Flask Server
```
python app.py
```
Akses Dashboard
Buka browser dan kunjungi:http://localhost:5000

## 📖 API Documentation
Aplikasi ini menyediakan REST API untuk integrasi frontend:
MethodEndpointDeskripsi
** GET/api/status Mengambil status counting mobil (FPS, Total Count).
** POST/api/start Memulai proses AI detection mobil.
** POST/api/stop Menghentikan proses AI detection mobil.
** GET/api/motor/status Mengambil status counting motor.
** GET/api/vehicle-reports Mengambil data lengkap laporan untuk grafik & tabel.
** POST/api/run_evaluation Menjalankan evaluasi model (YOLO vs SSD vs Ensemble).

## 📂 Struktur Folder
```
├── app.py                  # Entry point aplikasi Flask
├── ai_engine.py            # Core logic deteksi & tracking (VehicleDetector)
├── ai_evaluator.py         # Modul evaluasi model (ModelEvaluator)
├── video/                  # Folder simpan file video testing
├── dataset/                # Folder dataset gambar & label
├── templates/              # File HTML (Jinja2)
│   ├── dashboard_home.html
│   ├── antrian_mobil.html
│   ├── antrian_motor.html
│   └── evaluasi.html
└── static/                 # CSS, JS, Images
```

## 🤝 KontribusiPull requests dipersilakan. 
Untuk perubahan besar, harap buka issue terlebih dahulu untuk mendiskusikan apa yang ingin Anda ubah. 
* Fork project ini
* Buat feature branch (git checkout -b feature/AmazingFeature)
* Commit perubahan Anda (git commit -m 'Add some AmazingFeature')
* Push ke branch (git push origin feature/AmazingFeature)
* Buka Pull Request

## 📝 LicenseDistributed under the MIT License. 
See LICENSE for more information.
