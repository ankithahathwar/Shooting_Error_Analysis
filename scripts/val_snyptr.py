import torch
import intel_extension_for_pytorch as ipex 
from ultralytics import YOLO
import ultralytics.utils.torch_utils as torch_utils
import ultralytics.engine.validator as validator

# --- 🚀 THE FINAL SUPER PATCH ---
xpu_device = torch.device('xpu')

def force_xpu(*args, **kwargs):
    return xpu_device

# We must patch the validator's internal reference specifically
torch_utils.select_device = force_xpu
validator.select_device = force_xpu
# --------------------------------

def run_evaluation():
    # 1. Load your 91% accurate brain
    model = YOLO(r'models\snyptr_v1_91acc.pt')

    print(f"--- 📊 STARTING SCIENTIFIC EVALUATION ON TEST SET ---")

    # 2. Run Validation on the TEST split
    # This will compare predictions against your ground-truth labels
    metrics = model.val(
      data=r'C:\Users\Ankitha Hathwar\OneDrive\Documents\Snypter\Shooting_Error_Analysis\data.yaml', 
      split='test', 
      device=xpu_device
)

    # 3. Print the "B.Tech Project Gold"
    print("\n" + "="*30)
    print(f"🎯 FINAL TEST PRECISION: {metrics.results_dict['metrics/precision(B)']:.4f}")
    print(f"🎯 FINAL TEST RECALL:    {metrics.results_dict['metrics/recall(B)']:.4f}")
    print(f"🎯 FINAL TEST mAP50:     {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print("="*30)
    print("Check 'runs/detect/val/' for your Confusion Matrix and PR Curves!")

if __name__ == "__main__":
    run_evaluation()