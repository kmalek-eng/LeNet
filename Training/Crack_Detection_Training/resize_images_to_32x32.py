from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

SRC_DIR = BASE_DIR / "data" / "original_crack_dataset"
DEST_DIR = BASE_DIR / "data" / "cracks_32x32"
TARGET_SIZE = (32, 32)


def find_and_resize_images(src_dir, dest_dir, target_size=TARGET_SIZE):
    if not src_dir.exists():
        raise FileNotFoundError(f"Input dataset folder not found: {src_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    for root, _, files in src_dir.walk():
        relative_path = root.relative_to(src_dir)
        dest_sub_dir = dest_dir / relative_path
        dest_sub_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                src_file_path = root / file
                dest_file_path = dest_sub_dir / file

                try:
                    with Image.open(src_file_path) as img:
                        img = img.convert("RGB")
                        resized_img = img.resize(target_size)
                        resized_img.save(dest_file_path)
                    print(f"Saved: {dest_file_path}")
                except Exception as e:
                    print(f"Failed: {src_file_path} | {e}")


if __name__ == "__main__":
    find_and_resize_images(SRC_DIR, DEST_DIR)
