import numpy as np
import cv2
import json
import os
import random
from sklearn.neighbors import KernelDensity

# --- CONFIG ---
IMG_SIZE = 993  
IMAGES_PER_CLASS = 500  
SHOT_RADIUS = 12        
CLASS_ID = 1  
JSON_NAME = "C2_frontsight_dip_s"

def mm_to_px(x_mm, y_mm):
    px_x = int(((x_mm + 85) / 170) * IMG_SIZE)
    px_y = int((85 - y_mm) / 170 * IMG_SIZE)
    return px_x, px_y

def run_fix():
    json_path = f"seeds/{JSON_NAME}.json"
    if not os.path.exists(json_path): json_path += ".txt"
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    clean_target = cv2.imread('seeds/clean_target.png')
    # We still keep the target image to draw on
    
    print(f"🚀 Generating 500 VARIATED images for {JSON_NAME}...")

    for n in range(IMAGES_PER_CLASS):
        img = clean_target.copy()
        label_name = f"{JSON_NAME}_v_{n}"
        f_label = open(f"output/labels/{label_name}.txt", "w")
        
        # --- NATURAL VARIATION LOGIC ---
        # 1. Randomly pick a "center" for this specific image's cluster
        # This prevents every image from being exactly in the same spot
        group_center_x = random.uniform(-12, 12)
        group_center_y = random.uniform(-25, -15) # Stays in the LOW (C2) zone
        
        # 2. Pick a "Tightness" (How spread out the shots are)
        # 0.5 is a tight pro group, 2.5 is a messy beginner group
        tightness = random.uniform(0.6, 2.2)

        for _ in range(10): # 10 shots per target
            # 3. Create a natural "blob" around that group center
            # We add random "noise" to each shot to make it look like real shooting
            x_s = group_center_x + (random.uniform(-7, 7) * tightness)
            y_s = group_center_y + (random.uniform(-7, 7) * tightness)

            px, py = mm_to_px(x_s, y_s)

            if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
                # Draw the shot
                cv2.circle(img, (px, py), SHOT_RADIUS, (255, 255, 255), -1)
                cv2.circle(img, (px, py), SHOT_RADIUS, (0, 0, 0), 2)
                
                # Write YOLO Label (Normalized 0.0 - 1.0)
                f_label.write(f"{CLASS_ID} {px/IMG_SIZE} {py/IMG_SIZE} {(SHOT_RADIUS*2)/IMG_SIZE} {(SHOT_RADIUS*2)/IMG_SIZE}\n")
        
        f_label.close()
        cv2.imwrite(f"output/images/{label_name}.png", img)

if __name__ == "__main__":
    # Ensure folders exist
    os.makedirs("output/images", exist_ok=True)
    os.makedirs("output/labels", exist_ok=True)
    run_fix()
    print("\n✅ DONE! C2 now has natural, varied, low-positioned shot groups.")

    # The c2 codde in the generator.py was not working because the JSON data for C2 had a strong left bias and a low Y position. This fix script overrides those values to ensure the generated images have the correct "front sight dip" pattern, with shots centered around X=0 and Y=-20mm, while still preserving some variability in the Y direction.