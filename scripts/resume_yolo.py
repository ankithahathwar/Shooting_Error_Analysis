import torch
import intel_extension_for_pytorch as ipex 
from ultralytics import YOLO
import ultralytics.engine.trainer as trainer 
import ultralytics.utils.checks as checks

# --- 🚀 THE STABILITY PATCHES (Essential for Intel Arc) ---
# Forces the trainer to use XPU
trainer.select_device = lambda *args, **kwargs: torch.device('xpu')

# DISABLE the memory clearing function that causes CUDA errors
trainer.BaseTrainer._clear_memory = lambda *args, **kwargs: None

# Disable Mixed Precision (AMP) for Intel stability
checks.check_amp = lambda *args, **kwargs: False 
# ---------------------------------------------------------

def resume_training():
    # 🔍 TRIPLE CHECK THIS PATH: 
    # Open your 'runs/detect' folder to make sure it is 'snyptr_arc_final'
    checkpoint = 'runs/detect/snyptr_arc_final/weights/last.pt'
    
    # Load the checkpoint
    model = YOLO(checkpoint)

    print(f"--- 🎯 Resuming SNYPTR from {checkpoint} ---")

    # Resume the marathon
    # Note: resume=True handles everything (epochs, batch, workers) 
    # based on the original run's settings.
    model.train(resume=True)

if __name__ == "__main__":
    resume_training()