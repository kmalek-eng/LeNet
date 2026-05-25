import os
from PIL import Image

def find_and_resize_images(src_dir, dest_dir, target_size=(32, 32)):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    for root, _, files in os.walk(src_dir):
        # Construct the corresponding destination directory
        relative_path = os.path.relpath(root, src_dir)
        dest_sub_dir = os.path.join(dest_dir, relative_path)
        os.makedirs(dest_sub_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith('.jpg'):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_sub_dir, file)

                try:
                    with Image.open(src_file_path) as img:
                        resized_img = img.resize(target_size)
                        resized_img.save(dest_file_path)
                    print(f"Resized and saved {src_file_path} to {dest_file_path}")
                except Exception as e:
                    print(f"Failed to process {src_file_path}: {e}")

if __name__ == "__main__":
    src_directory = r"C:\!\Dessertation\Technical\AI\after_last\ImmerseNet\transfer\data\training"
    dest_directory = r"C:\!\Dessertation\Technical\AI\after_last\ImmerseNet\cracks\32_32"
    find_and_resize_images(src_directory, dest_directory)
