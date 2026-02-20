import os
import io
import numpy as np
import json
import multiprocessing as mp
from datetime import datetime
import warnings
import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = 20000000
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm.auto import tqdm
import pyarrow.parquet as pq
import shutil
import imagehash
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. Configuration ---
# Set workers to CPU count // 2 to avoid OOM and Pipe pressure
NUM_WORKERS = max(1, mp.cpu_count() - 1)
BATCH_SIZE = 1024  # Larger batch size for simple hashing
HASH_SIZE = 8      # 8x8 hash = 64 features

# IMPORTANT: Set this at the top level
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# --- 2. Dataset Class for Disk Loading ---
def compute_phash(args):
    """Worker function to compute pHash for a single image."""
    image_root, rel_path = args
    img_path = os.path.join(image_root, rel_path)
    try:
        with Image.open(img_path) as img:
            # Generate pHash and convert to flat boolean array (64 bits)
            hash_obj = imagehash.phash(img, hash_size=HASH_SIZE)
            return hash_obj.hash.flatten().astype(np.float32), rel_path, True
    except Exception:
        return np.zeros(HASH_SIZE * HASH_SIZE, dtype=np.float32), rel_path, False

def save_single_image(args):
    """Worker function to save a single image to disk."""
    img_bytes, rel_path, output_dir = args
    if img_bytes is None:
        return False
    try:
        save_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # If it's a dict (HF format), get bytes
        if isinstance(img_bytes, dict) and 'bytes' in img_bytes:
            img_bytes = img_bytes['bytes']

        with open(save_path, 'wb') as f:
            f.write(img_bytes)
        return True
    except Exception as e:
        print(f"Error saving {rel_path}: {e}")
        return False

# --- 3. Helper Functions ---

# --- 4. Main Processing Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate embeddings (hashes) from Parquet or local images")
    parser.add_argument("--repo", type=str, default="daominhwysi/toanmath.com-full", help="HF Repo ID (for parquet mode)")
    parser.add_argument("--image-dir", type=str, help="Local directory containing images (local mode)")
    parser.add_argument("--output-dir", type=str, default="data/toanmath_embeddings", help="Base directory for embeddings")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--limit", type=int, help="Limit number of files/images to process")

    args = parser.parse_args()

    # Paths setup
    embed_dir = os.path.join(args.output_dir, 'embeddings')
    progress_file = os.path.join(args.output_dir, 'progress.json')
    local_temp_dir = os.path.join(os.getcwd(), 'temp_data')
    os.makedirs(embed_dir, exist_ok=True)

    if args.image_dir:
        # Local Image mode
        process_local_images(args, embed_dir)
    else:
        # HF Parquet mode
        process_hf_repo(args, embed_dir, progress_file, local_temp_dir)

def process_local_images(args, embed_dir):
    """Processes images from a local directory."""
    image_paths = []
    print(f"Scanning {args.image_dir} for images...")
    for root, _, files in os.walk(args.image_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_paths.append(os.path.relpath(os.path.join(root, f), args.image_dir))

    if args.limit:
        image_paths = image_paths[:args.limit]

    print(f"Found {len(image_paths)} images.")

    all_embeddings = []
    valid_paths = []

    tasks = [(args.image_dir, path) for path in image_paths]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for features, path, success in tqdm(executor.map(compute_phash, tasks),
                                           total=len(tasks), desc="Generating hashes"):
            if success:
                all_embeddings.append(features)
                valid_paths.append(path)

    if all_embeddings:
        save_path = os.path.join(embed_dir, "local_images.npz")
        np.savez_compressed(
            save_path,
            embeddings=np.vstack(all_embeddings),
            paths=np.array(valid_paths)
        )
        print(f"Saved embeddings to {save_path}")

def process_hf_repo(args, embed_dir, progress_file, local_temp_dir):
    """Processes parquets from a HuggingFace repo (extracting to images first)."""
    # Closure to handle global-like progress file
    def load_prog():
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                return json.load(f)
        return {"processed_files": [], "last_updated": ""}

    def save_prog(filename, progress_data):
        if filename not in progress_data["processed_files"]:
            progress_data["processed_files"].append(filename)
        progress_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=4)

    all_files = list_repo_files(repo_id=args.repo, repo_type="dataset")
    parquet_files = sorted([f for f in all_files if f.endswith('.parquet')])
    progress = load_prog()
    files_to_process = [f for f in parquet_files if f not in progress["processed_files"]]

    if args.limit:
        files_to_process = files_to_process[:args.limit]

    for filename in tqdm(files_to_process, desc="Files"):
        save_path = os.path.join(embed_dir, filename.replace('.parquet', '.npz'))
        temp_image_dir = os.path.join(local_temp_dir, "extracted_images", filename.replace('.parquet', ''))

        try:
            downloaded_path = hf_hub_download(
                repo_id=args.repo, filename=filename,
                repo_type="dataset", local_dir=local_temp_dir
            )

            print(f"Extracting images from {filename}...")
            parquet_file = pq.ParquetFile(downloaded_path)
            all_paths = []

            # Step 1: Extract all images to disk first (Safer approach)
            os.makedirs(temp_image_dir, exist_ok=True)
            num_batches = (parquet_file.metadata.num_rows // 1024) + 1
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for batch in tqdm(parquet_file.iter_batches(batch_size=1024),
                                 total=num_batches, desc=f"Extracting {filename}", leave=False):
                    batch_dict = batch.to_pydict()
                    images = batch_dict.get('image', [])
                    paths = batch_dict.get('path', [])

                    tasks = [(img, path, temp_image_dir) for img, path in zip(images, paths)]
                    list(executor.map(save_single_image, tasks))
                    all_paths.extend(paths)

            # Step 2: Generate hashes from disk
            tqdm.write(f"Generating hashes for {len(all_paths)} images...")

            all_embeddings = []
            valid_paths = []

            tasks = [(temp_image_dir, path) for path in all_paths]

            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for features, path, success in tqdm(executor.map(compute_phash, tasks),
                                                   total=len(tasks), desc="  → Hashing", leave=False):
                    if success:
                        all_embeddings.append(features)
                        valid_paths.append(path)

            if all_embeddings:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez_compressed(
                    save_path,
                    embeddings=np.vstack(all_embeddings),
                    paths=np.array(valid_paths)
                )
                save_prog(filename, progress)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(temp_image_dir):
                shutil.rmtree(temp_image_dir)
            if 'downloaded_path' in locals() and os.path.exists(downloaded_path):
                os.remove(downloaded_path)

if __name__ == "__main__":
    main()
