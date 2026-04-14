import os

# Do this for 'train/labels', 'valid/labels', and 'test/labels'
folders = [r'Shooting_Dataset\train\labels', r'Shooting_Dataset\valid\labels', r'Shooting_Dataset\test\labels']

for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r") as f:
                lines = f.readlines()
            
            # Keep only lines where the class (first number) is 0, 1, 2, or 3
            clean_lines = [l for l in lines if int(l.split()[0]) < 4]
            
            with open(path, "w") as f:
                f.writelines(clean_lines)
print("✅ Dataset Scrubbed! No more 'Corrupt' errors.")