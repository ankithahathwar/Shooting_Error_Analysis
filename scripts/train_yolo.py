import torch
import intel_extension_for_pytorch as ipex 
from ultralytics import YOLO
import ultralytics.engine.trainer as trainer 
import ultralytics.utils.checks as checks

# --- 🚀 THE ULTIMATE INTEL ARC PATCHES ---
# 1. Force the trainer to use XPU
trainer.select_device = lambda *args, **kwargs: torch.device('xpu')

# 2. DISABLE the memory clearing function that uses CUDA commands
# This is what caused your "AssertionError: Torch not compiled with CUDA"
trainer.BaseTrainer._clear_memory = lambda *args, **kwargs: None

# 3. Disable Mixed Precision (AMP) for stability
checks.check_amp = lambda *args, **kwargs: False 
# -----------------------------------------

def train_snyptr():
    model = YOLO('yolov8n.pt')

    print("--- 🎯 FINAL ATTEMPT: Stability Mode Active ---")

    model.train(
        data='Shooting_Dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=4,           # Keep it at 4 for safety
        workers=0,         # Keep at 0 to save RAM
        device='xpu',
        amp=False,
        optimizer='AdamW',
        name='snyptr_arc_final'
    )

if __name__ == "__main__":
    train_snyptr()

