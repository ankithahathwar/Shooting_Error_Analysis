import torch
import intel_extension_for_pytorch as ipex 
from ultralytics import YOLO
import ultralytics.utils.torch_utils as torch_utils
import ultralytics.engine.trainer as trainer
import ultralytics.utils.checks as checks

# --- 🚀 THE STABILITY PATCH (V4: GRADIENT SHIELD) ---
xpu_device = torch.device('xpu')
def force_xpu(*args, **kwargs): return xpu_device

torch_utils.select_device = force_xpu
trainer.select_device = force_xpu

def disabled_check_amp(*args, **kwargs): return False
checks.check_amp = disabled_check_amp
trainer.check_amp = disabled_check_amp

def fake_get_memory(self, fraction=False): return 0.1
trainer.BaseTrainer._get_memory = fake_get_memory
# -----------------------------------------------------

def run_booster():
    # Load your ORIGINAL 12-hour brain again (v1)
    model = YOLO(r'models\snyptr_v1_91acc.pt')

    print(f"--- 🚀 RESTARTING WITH STABILITY LIMITERS ---")

    model.train(
        data='data.yaml',
        epochs=10,
        imgsz=640,
        device=xpu_device, 
        # 🛡️ THE STABILITY TWEAKS:
        lr0=0.0001,        # 5x lower than before (much safer)
        warmup_epochs=0,    # Skip warmup (helps prevent nan in fine-tuning)
        batch=4,           
        workers=0,         
        amp=False,
        mosaic=1.0,         
        exist_ok=True,
        plots=True         # We want to see the charts in the morning!
    )

if __name__ == "__main__":
    run_booster()