import os
import wget
import tarfile
from roboflow import Roboflow
import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# ===============================
# 1. KONFIGURASI USER
# ===============================

ROBOFLOW_API_KEY = "UHQQz4I0qx4z5ba9qHH4"
ROBOFLOW_WORKSPACE = "gree"
ROBOFLOW_PROJECT = "roro-vehicle-detection"
ROBOFLOW_VERSION = 1

NUM_CLASSES = 5       # WAJIB sama dengan label_map
BATCH_SIZE = 2         # LAPTOP KENTANG MODE
NUM_STEPS = 30000      # Lebih ringan dari 50k

# ===============================
# 2. DOWNLOAD PRETRAINED MODEL
# ===============================

model_name = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"
model_url = f"http://download.tensorflow.org/models/object_detection/tf2/20200711/{model_name}.tar.gz"
pretrained_dir = "pre_trained_model"

os.makedirs(pretrained_dir, exist_ok=True)

if not os.path.exists(os.path.join(pretrained_dir, model_name)):
    print("Downloading pretrained model...")
    wget.download(model_url)
    print("\nExtracting...")
    with tarfile.open(f"{model_name}.tar.gz") as tar:
        tar.extractall(pretrained_dir)

# ===============================
# 3. DOWNLOAD DATASET ROBOFLOW
# ===============================

print("\nDownloading dataset from Roboflow...")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
dataset = project.version(ROBOFLOW_VERSION).download("tfrecord")

dataset_dir = dataset.location
train_record = os.path.join(dataset_dir, "train", "vehicle.tfrecord")
valid_record = os.path.join(dataset_dir, "valid", "vehicle.tfrecord")
label_map = os.path.join(dataset_dir, "train", "vehicle_label_map.pbtxt")

# ===============================
# 4. UPDATE PIPELINE.CONFIG
# ===============================

pipeline_path = os.path.join(pretrained_dir, model_name, "pipeline.config")

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(pipeline_path, "r") as f:
    text_format.Merge(f.read(), pipeline_config)

# --- Model ---
pipeline_config.model.ssd.num_classes = NUM_CLASSES

# --- Training ---
pipeline_config.train_config.batch_size = BATCH_SIZE
pipeline_config.train_config.num_steps = NUM_STEPS
pipeline_config.train_config.sync_replicas = False

pipeline_config.train_config.fine_tune_checkpoint = os.path.join(
    pretrained_dir, model_name, "checkpoint", "ckpt-0"
)
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"

# --- LEARNING RATE KENTANG ---
pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = 0.005
pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_learning_rate = 0.001

# --- Input ---
pipeline_config.train_input_reader.label_map_path = label_map
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [train_record]

pipeline_config.eval_input_reader[0].label_map_path = label_map
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [valid_record]

# --- Save ---
with tf.io.gfile.GFile("pipeline.config", "w") as f:
    f.write(text_format.MessageToString(pipeline_config))

print("\n✅ SIAP!")
print("pipeline.config sudah dioptimalkan")
print("Batch size:", BATCH_SIZE)
print("Num steps:", NUM_STEPS)
