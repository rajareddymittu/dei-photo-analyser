import os
import shutil

# Path to your main folder containing the subfolders
SOURCE_DIR = "./dataset_1000_photos_"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

counter = 1

# Iterate through all subdirectories
for root, dirs, files in os.walk(SOURCE_DIR):
    # Skip the root directory itself to only process files inside subfolders
    if root == SOURCE_DIR:
        continue

    for filename in sorted(files):
        ext = os.path.splitext(filename)[1].lower()
        if ext in VALID_EXTENSIONS:
            src_path = os.path.join(root, filename)
            
            # Destination path: main_folder/brand_1.jpg, brand_2.jpg, ...
            dest_name = f"brand_{counter}{ext}"
            dest_path = os.path.join(SOURCE_DIR, dest_name)

            shutil.move(src_path, dest_path)
            counter += 1

# Remove the empty subfolders
for folder in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder)
    if os.path.isdir(folder_path):
        try:
            os.rmdir(folder_path)
        except OSError:
            # In case any non-image file remained
            shutil.rmtree(folder_path)

print(f"Done! Moved and renamed {counter - 1} images to '{SOURCE_DIR}'.")
