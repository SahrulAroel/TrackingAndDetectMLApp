import onnxruntime as ort

# Ganti path sesuai lokasi model Anda
model_path = 'WeightModel/WeightBaru/model.onnx'

try:
    session = ort.InferenceSession(model_path)
    
    print("\n=== INFORMASI INPUT MODEL ===")
    for i in session.get_inputs():
        print(f"Name: {i.name}")
        print(f"Shape: {i.shape}")
        print(f"Type: {i.type}")
        print("-" * 20)

    print("\n=== INFORMASI OUTPUT MODEL ===")
    for o in session.get_outputs():
        print(f"Name: {o.name}")
        print(f"Shape: {o.shape}")
        print("-" * 20)
        
except Exception as e:
    print(f"Error: {e}")