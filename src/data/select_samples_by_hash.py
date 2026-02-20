import os
import numpy as np
import argparse
import json
import shutil
import glob
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from PIL import Image
import io
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def get_hex_from_hash(h):
    """Converts a flattened 64-dim binary hash (as float) to a hex string for sorting."""
    # Convert float/bool to bits
    bits = (h > 0.5).astype(int)
    # Pack 8 bits into a byte
    bytes_list = []
    for i in range(0, 64, 8):
        byte_val = 0
        for b in range(8):
            byte_val = (byte_val << 1) | bits[i + b]
        bytes_list.append(byte_val)
    return bytes(bytes_list).hex()

def save_image_worker(args):
    """Worker function to save a single image."""
    img_bytes, rel_path, output_dir = args
    if img_bytes is None: return False
    try:
        save_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if isinstance(img_bytes, dict) and 'bytes' in img_bytes:
            img_bytes = img_bytes['bytes']
        with open(save_path, 'wb') as f:
            f.write(img_bytes)
        return True
    except Exception as e:
        print(f"Error saving {rel_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Sort images by hash and select top N samples")
    parser.add_argument("--embed-dir", type=str, default="data/toanmath_embeddings/embeddings", help="Directory with .npz files")
    parser.add_argument("--repo", type=str, default="daominhwysi/toanmath.com-full", help="HF Repo ID for downloading parquets")
    parser.add_argument("--output-dir", type=str, default="data/selected_samples_25k", help="Where to save selected images")
    parser.add_argument("--n", type=int, default=25000, help="Number of images to select")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))

    args = parser.parse_args()

    # 1. Collect all hashes
    print(f"Loading hashes from {args.embed_dir}...")
    npz_pattern = os.path.join(args.embed_dir, "*.npz")
    npz_files = [f for f in glob.glob(npz_pattern) if os.path.basename(f) != "local_images.npz"]

    all_items = [] # Will store (hex_hash, parquet_filename, rel_path)

    for npz_path in tqdm(npz_files, desc="Reading NPZ files"):
        data = np.load(npz_path)
        hashes = data['embeddings']
        paths = data['paths']
        parquet_filename = os.path.basename(npz_path).replace('.npz', '.parquet')

        for h, p in zip(hashes, paths):
            hex_h = get_hex_from_hash(h)
            all_items.append((hex_h, parquet_filename, p))

    print(f"Found total {len(all_items)} images.")

    # 2. Sort by hash
    print("Sorting by hash...")
    all_items.sort(key=lambda x: x[0])

    # 3. Select top N
    selected_items = all_items[:args.n]
    print(f"Selected {len(selected_items)} images.")

    # 4. Group by parquet for efficient extraction
    parquet_groups = {}
    for hex_h, parquet_fn, rel_path in selected_items:
        if parquet_fn not in parquet_groups:
            parquet_groups[parquet_fn] = []
        parquet_groups[parquet_fn].append(rel_path)

    # 5. Extract images
    os.makedirs(args.output_dir, exist_ok=True)
    temp_download_dir = "temp_parquets"
    os.makedirs(temp_download_dir, exist_ok=True)

    try:
        for parquet_fn, rel_paths in tqdm(parquet_groups.items(), desc="Extracting from Parquets"):
            print(f"\nProcessing {parquet_fn} ({len(rel_paths)} images)...")

            # Download parquet
            downloaded_path = hf_hub_download(
                repo_id=args.repo, filename=parquet_fn,
                repo_type="dataset", local_dir=temp_download_dir
            )

            # Read parquet
            parquet_file = pq.ParquetFile(downloaded_path)
            path_set = set(rel_paths)

            extracted_count = 0
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                num_batches = (parquet_file.metadata.num_rows // 1024) + 1
                for batch in tqdm(parquet_file.iter_batches(batch_size=1024),
                                 total=num_batches, desc=f"  Scanning {parquet_fn}", leave=False):
                    batch_dict = batch.to_pydict()
                    images = batch_dict.get('image', [])
                    paths = batch_dict.get('path', [])

                    tasks = []
                    for img, path in zip(images, paths):
                        if path in path_set:
                            tasks.append((img, path, args.output_dir))

                    if tasks:
                        results = list(executor.map(save_image_worker, tasks))
                        extracted_count += sum(1 for r in results if r)

                    if extracted_count >= len(rel_paths):
                        break

            print(f"  Extracted {extracted_count} images.")

            # Cleanup parquet to save space
            if os.path.exists(downloaded_path):
                os.remove(downloaded_path)

    finally:
        if os.path.exists(temp_download_dir):
            shutil.rmtree(temp_download_dir)

if __name__ == "__main__":
    main()
