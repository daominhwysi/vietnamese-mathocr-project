print("1. Loading OS/Pathlib...", flush=True)
import os
import shutil
from pathlib import Path

print("2. Loading PIL/TQDM...", flush=True)
from PIL import Image
from tqdm import tqdm

print("3. Loading Torch (this usually hangs)...", flush=True)
import torch

print("4. Loading Transformers...", flush=True)
from transformers import pipeline

print("5. All imports finished!", flush=True)

def flatten_images(base_dir):
    """
    Flattens the directory structure of selected_samples_25k/images.
    Moves all images from subdirectories directly into the images folder.
    Prefixes names with subdirectory names to avoid collisions.
    """
    images_dir = Path(base_dir) / "images"
    if not images_dir.exists():
        print(f"Directory {images_dir} does not exist.")
        return

    subdirs = [d for d in images_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"Directory {images_dir} is already flattened. Skipping.")
        return

    print(f"Found {len(subdirs)} subdirectories to flatten...")
    for subdir in tqdm(subdirs, desc="Flattening"):
        for img_path in subdir.iterdir():
            if img_path.is_file():
                # Create a new unique name using the subdirectory name as a prefix
                new_name = f"{subdir.name}_{img_path.name}"
                target_path = images_dir / new_name
                shutil.move(str(img_path), str(target_path))

        # Remove empty subdirectory
        try:
            subdir.rmdir()
        except OSError:
            print(f"Warning: Could not remove directory {subdir}. It might not be empty.")

def run_layout_analysis(base_dir):
    """
    Uses DocLayout model to annotate images and save in YOLO format.
    """
    images_dir = Path(base_dir) / "images"
    labels_dir = Path(base_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]

    # Skip already processed images
    image_files = [f for f in image_files if not (labels_dir / f"{f.stem}.txt").exists()]

    if not image_files:
        print("All images have already been processed.")
        return

    print("Loading model...")
    device = 0 if torch.cuda.is_available() else -1
    layout_detector = pipeline(
        "object-detection",
        model="PaddlePaddle/PP-DocLayoutV3_safetensors",
        device=device
    )

    # Label mapping (keeping the raw IDs from the model)
    id2label = layout_detector.model.config.id2label
    print(f"Model ID to Label mapping: {id2label}")

    print(f"Found {len(image_files)} images to process.")

    # Process in batches for better performance
    batch_size = 8
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing Batches"):
        batch_paths = image_files[i : i + batch_size]
        batch_images = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"Error opening {p}: {e}")

        if not batch_images:
            continue

        try:
            results = layout_detector(batch_images)
        except Exception as e:
            print(f"Error during detection: {e}")
            continue

        for img_path, img_results, img_obj in zip(valid_paths, results, batch_images):
            img_width, img_height = img_obj.size
            label_file = labels_dir / f"{img_path.stem}.txt"

            with open(label_file, "w") as f:
                for res in img_results:
                    # YOLO format: class_id x_center y_center width height (normalized)
                    box = res["box"]
                    xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

                    # Ensure coordinates are within image bounds
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_width, xmax)
                    ymax = min(img_height, ymax)

                    width = xmax - xmin
                    height = ymax - ymin
                    x_center = xmin + width / 2
                    y_center = ymin + height / 2

                    # Normalize
                    x_center /= img_width
                    y_center /= img_height
                    width /= img_width
                    height /= img_height

                    label_id = -1
                    # Extract numeric ID from label name if needed,
                    # but pipeline usually returns the predicted class index in the dict if accessed correctly.
                    # Or we can find the ID from id2label.
                    label_name = res["label"]
                    # Find the ID for this label name. Note: some names repeat,
                    # we'll use the first one that matches or the model might return ID directly.
                    # Actually, Transformers pipeline object-detection returns 'label' as the string name.
                    for idx, name in id2label.items():
                        if name == label_name:
                            label_id = idx
                            break

                    if label_id != -1:
                        f.write(f"{label_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Create a simple data.yaml for information
    with open(Path(base_dir) / "data.yaml", "w") as f:
        f.write("names:\n")
        # Ensure we write all possible IDs
        max_id = max(id2label.keys())
        for idx in range(max_id + 1):
            name = id2label.get(idx, f"class_{idx}")
            f.write(f"  {idx}: {name}\n")

if __name__ == "__main__":
    DATA_DIR = "data/selected_samples_25k"
    print("Step 1: Flatten")
    # 
    flatten_images(DATA_DIR)
    print("Step 2: Annotate")

    # 
    run_layout_analysis(DATA_DIR)
