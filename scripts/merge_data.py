import os
import shutil
import random

# --- CONFIG ---
BASE_PATH = r"C:\Users\Ankitha Hathwar\OneDrive\Documents\Snypter\Shooting_Error_Analysis"
SYNTH_IMG_DIR = os.path.join(BASE_PATH, "output", "images")
SYNTH_LBL_DIR = os.path.join(BASE_PATH, "output", "labels")
DEST_DIR = os.path.join(BASE_PATH, "Shooting_Dataset")

SPLIT_RATIO = 0.8  # 80% to Train, 20% to Valid

def split_and_merge():
    # 1. Get all synthetic images
    if not os.path.exists(SYNTH_IMG_DIR):
        print(f"❌ Error: Can't find synthetic images at {SYNTH_IMG_DIR}")
        return
    
    all_images = [f for f in os.listdir(SYNTH_IMG_DIR) if f.lower().endswith(('.png', '.jpg'))]
    random.shuffle(all_images) # Shuffle to mix classes
    
    split_idx = int(len(all_images) * SPLIT_RATIO)
    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

    def move_files(files, target_folder):
        count = 0
        img_dest = os.path.join(DEST_DIR, target_folder, "images")
        lbl_dest = os.path.join(DEST_DIR, target_folder, "labels")
        
        # Ensure destination folders exist
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        for img_file in files:
            # Move Image
            shutil.copy(os.path.join(SYNTH_IMG_DIR, img_file), os.path.join(img_dest, img_file))
            
            # Move corresponding Label
            lbl_file = os.path.splitext(img_file)[0] + ".txt"
            src_lbl = os.path.join(SYNTH_LBL_DIR, lbl_file)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(lbl_dest, lbl_file))
                count += 1
        return count

    print(f"📦 Found {len(all_images)} synthetic images. Starting split...")
    
    # 2. Execute the move
    t_count = move_files(train_files, "train")
    v_count = move_files(val_files, "valid") # Roboflow uses 'valid' folder name

    print(f"\n✅ DONE!")
    print(f"🚀 Sent {t_count} pairs to Shooting_Dataset/train")
    print(f"🧪 Sent {v_count} pairs to Shooting_Dataset/valid")
    print(f"⚠️ Test folder was left 100% real for your research paper accuracy!")

if __name__ == "__main__":
    split_and_merge()