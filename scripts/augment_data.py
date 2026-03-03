import albumentations as A
import cv2
import os

# 1. Setup absolute paths to prevent "FileNotFound" errors
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5), 
    A.Rotate(limit=5, p=0.5),          
    A.GaussNoise(std_range=(0.1, 0.2), p=0.3), 
    A.Blur(blur_limit=3, p=0.2),       
])

def generate_bulk_data(class_mapping):
    for folder_name, multiplier in class_mapping.items():
        input_dir = os.path.join(BASE_DIR, 'Shooting_Dataset', folder_name)
        output_dir = os.path.join(BASE_DIR, 'Augmented_Data', folder_name)
        
        if not os.path.exists(input_dir):
            print(f"Skipping {folder_name}: Folder not found.")
            continue

        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)

        print(f"Augmenting {folder_name} (x{multiplier})...")
        for img_file in os.listdir(input_dir):
            image = cv2.imread(os.path.join(input_dir, img_file))
            if image is None: continue
            
            for i in range(multiplier):
                augmented = transform(image=image)['image']
                cv2.imwrite(os.path.join(output_dir, f"aug_{i}_{img_file}"), augmented)
    print("All classes processed successfully!")

# 2. Define exactly how many variations to make for each class
# If you have few images in one class, increase its multiplier!
my_classes = {
    "C1_to_and_fro": 100,
    "C2_frontsight_dip": 100,
    "C3_over-tight_grip": 100,
    "C4_breathe_control": 100,
    "C5_early_recoil": 100,
    "C6_stance": 100,
    "C7_acute_angle": 100
}

generate_bulk_data(my_classes)