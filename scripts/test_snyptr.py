import torch
import intel_extension_for_pytorch as ipex 
from ultralytics import YOLO
import ultralytics.utils.torch_utils as torch_utils
import ultralytics.engine.predictor as predictor

# --- 🛠️ THE SUPER SURGICAL PATCH ---
# 1. Create the device object once
xpu_device = torch.device('xpu')

# 2. Define a function that ALWAYS returns our Intel GPU
def force_xpu(*args, **kwargs):
    return xpu_device

# 3. Patch the function in TWO places to catch the local references
torch_utils.select_device = force_xpu
predictor.select_device = force_xpu
# -------------------------------------

def run_final_test():
    model_path = r'models\snyptr_v1_91acc.pt'
    test_data_path = r'Shooting_Dataset\test\images'

    model = YOLO(model_path)
    print(f"--- 🎯 SNYPTR FINAL TEST STARTING (SUPER PATCH ACTIVE) ---")

    # 🔥 BYPASS STRATEGY: 
    # Instead of the string 'xpu', we pass the actual device object.
    # YOLO's select_device function is programmed to return immediately 
    # if it sees a device object, skipping all the CUDA checks!
    results = model.predict(
        source=test_data_path,
        device=xpu_device, 
        save=True,           
        save_txt=True,       
        conf=0.2             
    )

    print(f"--- ✅ Test Complete! Check 'runs/detect/predict' ---")

if __name__ == "__main__":
    run_final_test()