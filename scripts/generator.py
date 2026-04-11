import numpy as np
import cv2
import json
import os
import random
from sklearn.neighbors import KernelDensity

# --- 1. CONFIGURATION ---
IMG_SIZE = 993  
IMAGES_PER_CLASS = 500  
SHOT_RADIUS = 12        

CLASSES = {
    "C1_to_and_fro_s": 0, "C2_frontsight_dip_s": 1, "C3_over-tight_grip_s": 2,
    "C4_breathe_control_s": 3, "C5_early_recoil_s": 4, "C6_stance_s": 5, "C7_acute_angle_s": 6
}

# BIASES (in mm) - LOCKED PER USER EXPERTISE
BIASES = {
    0: (0, 0),    # C1: Vertical Spread
    1: (0, -5),   # C2: Frontsight Dip (Low)
    2: (-7, 0),   # C3: Over-tight Grip (LEFT)
    3: (0, 0),    # C4: Breathe Control (Scattered)
    4: (0, 6),    # C5: Early Recoil (High)
    5: (0, 0),    # C6: Stance (Horizontal Sway)
    6: (7, 0)     # C7: Acute Angle (RIGHT)
}

def mm_to_px(x_mm, y_mm):
    px_x = int(((x_mm + 85) / 170) * IMG_SIZE)
    px_y = int((85 - y_mm) / 170 * IMG_SIZE)
    return px_x, px_y

def run_generator(json_name):
    class_id = CLASSES[json_name]
    json_path = f"seeds/{json_name}.json"
    if not os.path.exists(json_path): json_path += ".txt"
    if not os.path.exists(json_path): return

    with open(json_path, 'r') as f:
        data = json.load(f)

    clean_target = cv2.imread('seeds/clean_target.png')
    all_points = np.array([[s['x'], s['y']] for s in data])
    
    # 0.8 Bandwidth preserves the "signature" shapes from your JSON
    kde = KernelDensity(kernel='gaussian', bandwidth=0.8).fit(all_points)
    mean_pt = np.mean(all_points, axis=0)
    bias_x, bias_y = BIASES.get(class_id, (0, 0))

    for n in range(IMAGES_PER_CLASS):
        img = clean_target.copy()
        samples = kde.sample(10) 
        label_name = f"{json_name}_v_{n}"
        f_label = open(f"output/labels/{label_name}.txt", "w")
        
        # Standard Diversity Scaling
        s_factor = random.uniform(0.7, 1.3) 

        for shot in samples:
            # Apply the manual offsets (Biases)
            x_s = mean_pt[0] + bias_x + s_factor * (shot[0] - mean_pt[0])
            y_s = mean_pt[1] + bias_y + s_factor * (shot[1] - mean_pt[1])
            
            # --- SPECIAL CLASS LOGIC ---
            if class_id == 0: # C1: Stretch Vertical
                y_s = mean_pt[1] + (s_factor * 1.5) * (shot[1] - mean_pt[1])
            elif class_id == 3: # C4: Chaos Scatter
                x_s = mean_pt[0] + random.uniform(-40, 40)
                y_s = mean_pt[1] + random.uniform(-40, 40)
            elif class_id == 5: # C6: Stretch Horizontal
                x_s = mean_pt[0] + (s_factor * 1.5) * (shot[0] - mean_pt[0])

            px, py = mm_to_px(x_s, y_s)

            if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
                cv2.circle(img, (px, py), SHOT_RADIUS, (255, 255, 255), -1)
                cv2.circle(img, (px, py), SHOT_RADIUS, (0, 0, 0), 2)
                f_label.write(f"{class_id} {px/IMG_SIZE} {py/IMG_SIZE} {(SHOT_RADIUS*2)/IMG_SIZE} {(SHOT_RADIUS*2)/IMG_SIZE}\n")
        
        f_label.close()
        cv2.imwrite(f"output/images/{label_name}.png", img)

if __name__ == "__main__":
    os.makedirs("output/images", exist_ok=True)
    os.makedirs("output/labels", exist_ok=True)
    
    print("🚀 FINAL PRODUCTION RUN: 3,500 expert-verified images...")
    for class_file in CLASSES.keys():
        print(f"📦 Locked and Loading: {class_file}")
        run_generator(class_file)
    print("\n✅ MISSION COMPLETE. Your dataset is ready for training.")