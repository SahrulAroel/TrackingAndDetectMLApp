🚗 Vehicle Monitoring System (Flask + YOLOv8 + CUDA)
Sistem monitoring lalu lintas berbasis AI untuk deteksi dan klasifikasi golongan kendaraan secara real-time. Aplikasi ini menggunakan Flask sebagai backend dan mendukung akselerasi hardware NVIDIA CUDA untuk performa deteksi yang optimal.

✨ Fitur Utama
Dual Stream Monitoring: Pemisahan jalur deteksi untuk Mobil (Multi-Lane) dan Motor.

Klasifikasi Golongan: Deteksi otomatis berdasarkan Golongan II, IVA, IVB, dan VB.

AI Model Evaluator: Bandingkan performa model YOLOv8, SSD, dan Ensemble secara langsung dari dashboard.

Laporan & Statistik: Integrasi SQLite untuk menyimpan riwayat melintas dan visualisasi tren harian.

Hardware Accelerated: Dioptimalkan untuk penggunaan GPU NVIDIA (CUDA).

💻 Persyaratan Sistem
Python: 3.9 atau lebih baru.

OS: Windows 10/11 atau Linux (Ubuntu direkomendasikan).

GPU: NVIDIA GPU (RTX 20-series ke atas direkomendasikan).

Drivers: NVIDIA Driver (Versi terbaru) + CUDA Toolkit 11.8/12.1.

🚀 Panduan Instalasi (NVIDIA CUDA)
Ikuti langkah-langkah ini untuk memastikan aplikasi berjalan menggunakan GPU:

1. Persiapan Environment
Bash
# Clone repository ini
git clone https://github.com/username/project-name.git
cd project-name

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
2. Instalasi PyTorch dengan CUDA
Penting: Jangan hanya pip install torch. Gunakan perintah resmi dari PyTorch agar versi CUDA terdeteksi. Contoh untuk CUDA 11.8:

Bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
3. Instalasi Dependencies Lainnya
Bash
pip install -r requirements.txt
Pastikan requirements.txt mencakup: flask, flask-cors, opencv-python, ultralytics, numpy, pandas.

4. Konfigurasi Dataset & Model
Letakkan file model .pt di dalam folder WeightModel/.

Pastikan struktur folder dataset sesuai:

dataset/train/

dataset/valid/

dataset/test/

⚙️ Konfigurasi Database & Video
Ubah variabel berikut di dalam app.py jika lokasi file Anda berbeda:

Python
VIDEO_SOURCE_MOBIL = 'video/assa.mp4'
VIDEO_SOURCE_MOTOR = 'video/TestMotor.mp4'
# Path database default berada di D:/ atau folder root project
🛠️ Cara Menjalankan
Pastikan GPU Anda terdeteksi oleh sistem sebelum menjalankan aplikasi:

Bash
# Cek status CUDA (Opsional)
python -c "import torch; print(torch.cuda.is_available())"

# Jalankan aplikasi
python app.py
Buka browser dan akses: http://localhost:5000

📊 Struktur Folder
Plaintext
├── WeightModel/        # Menyimpan model YOLO (.pt)
├── video/              # Source video testing
├── static/             # File CSS, JS, dan Gambar UI
├── templates/          # File HTML (Halaman Dashboard)
├── ai_engine.py        # Logika utama deteksi & tracking
├── ai_evaluator.py     # Logika perbandingan model
├── app.py              # Flask server & API Endpoints
└── laporan_kendaraan.db # Database SQLite (Auto-generated)
🧪 Evaluasi Model
Halaman Evaluasi memungkinkan Anda menguji model terhadap dataset validasi.

YOLOv8: Cepat dan akurat (Recommended).

SSD: Ringan namun presisi lebih rendah.

Ensemble: Gabungan kedua model untuk hasil maksimal (Memerlukan VRAM lebih besar).

📝 Catatan Penting
Tracker Reset: Sistem akan mereset tracker secara otomatis setiap kali evaluasi selesai dilakukan untuk mencegah ID kendaraan yang "nyangkut".

CORS: Diaktifkan secara default untuk mempermudah integrasi dengan frontend modern (React/Vue).

Kontribusi: Jika Anda ingin mengembangkan fitur baru, silakan buat Pull Request atau buka Issue baru.
