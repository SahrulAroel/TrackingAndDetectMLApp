from ultralytics import YOLO

# Load model YOLOv8 kamu (bisa .pt hasil training)
model = YOLO("WeightModel/WeightBaru/best.pt") 

# Export ke ONNX
path = model.export(format="onnx", opset=17)
print(f"Model berhasil disimpan di: {path}")