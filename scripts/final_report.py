import torch
import intel_extension_for_pytorch as ipex
from ultralytics import YOLO
import ultralytics.utils.torch_utils as torch_utils
import ultralytics.utils.checks as checks

# --- 🚀 THE INTEL ARC STABILITY PATCH ---
xpu_device = torch.device('xpu')
def force_xpu(*args, **kwargs): return xpu_device
torch_utils.select_device = force_xpu
checks.check_amp = lambda *args: False 
# ----------------------------------------

def generate_comprehensive_report():
    # 1. Load your NEW best weights
    model_path = r'runs\detect\train\weights\best.pt'
    model = YOLO(model_path)
    
    print("--- 📈 STARTING COMPREHENSIVE EVALUATION ---")
    
    # 2. Run Validation with ALL plots enabled
    metrics = model.val(
        data='data.yaml',
        device=xpu_device,
        plots=True,          # Generates Confusion Matrix, PR Curve, F1 Curve
        save_json=True,       # Saves results to a JSON for further analysis
        split='test',         # Run this on your TEST set for the final report
        name='viva_evaluation' # Saves to runs/detect/viva_evaluation
    )
    
    # 3. Print the High-Level Summary
    print("\n" + "="*30)
    print("      OVERALL METRICS")
    print("="*30)
    print(f"Mean Precision: {metrics.results_dict['metrics/precision(B)']:.4f}")
    print(f"Mean Recall:    {metrics.results_dict['metrics/recall(B)']:.4f}")
    print(f"mAP @50:        {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP @50-95:     {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
    print("="*30)

    # 4. Print Per-Class Metrics (Crucial for showing you fixed C3!)
    print("\n" + "="*30)
    print("    PER-CLASS BREAKDOWN")
    print("="*30)
    names = model.names
    for i, class_name in names.items():
        # Accessing per-class metrics
        p = metrics.class_result(i)[0] # Precision
        r = metrics.class_result(i)[1] # Recall
        map50 = metrics.class_result(i)[2] # mAP50
        print(f"Class [{class_name}]: P={p:.4f}, R={r:.4f}, mAP50={map50:.4f}")
    print("="*30)

    print(f"\n✅ All graphs and the Confusion Matrix have been saved to:")
    print(f"📂 runs/detect/viva_evaluation/")

if __name__ == "__main__":
    generate_comprehensive_report()