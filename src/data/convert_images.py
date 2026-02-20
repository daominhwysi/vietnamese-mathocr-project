import os
import io
import argparse
import pyarrow.parquet as pq
import warnings
from PIL import Image
# Set a reasonable limit roughly 20MP (model only uses 256x256)
Image.MAX_IMAGE_PIXELS = 20000000
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
from tqdm.auto import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def save_image(args):
    """Worker function to save a single image."""
    img_bytes, rel_path, output_dir = args
    if img_bytes is None:
        return False

    try:
        # Construct absolute save path
        save_path = os.path.join(output_dir, rel_path)

        # Create subdirectories if they don't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # If it's a dict (HF format), extract bytes
        if isinstance(img_bytes, dict) and 'bytes' in img_bytes:
            img_bytes = img_bytes['bytes']

        # Validate image is not a "bomb" before saving
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                # Accessing size is safe, it only reads the header
                _ = img.size
        except Exception as e:
            # If it exceeds limit, it will raise DecompressionBombError
            print(f"Skipping potential bomb or corrupt image {rel_path}: {e}")
            return False

        # Write bytes directly to file
        with open(save_path, 'wb') as f:
            f.write(img_bytes)
        return True
    except Exception as e:
        print(f"Error saving {rel_path}: {e}")
        return False

def convert_parquet_to_images(parquet_path, output_dir, num_workers=None, limit=None):
    """
    Converts images stored in a Parquet file to individual image files using parallel processing.
    """
    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found at {parquet_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading Parquet file: {parquet_path}")
    parquet_file = pq.ParquetFile(parquet_path)

    # Use CPU count if num_workers not specified
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    total_rows = parquet_file.metadata.num_rows
    if limit:
        total_rows = min(total_rows, limit)

    print(f"Total images to process: {total_rows}")

    processed_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Process in batches to manage memory
        for batch in tqdm(parquet_file.iter_batches(batch_size=1024),
                         total=(total_rows // 1024) + 1,
                         desc="Processing batches"):

            if limit and processed_count >= limit:
                break

            batch_dict = batch.to_pydict()

            # Prepare tasks for parallel execution
            # Handling potential different column names or HF structures
            images = batch_dict.get('image', [])
            paths = batch_dict.get('path', [])

            # Slice batch if it exceeds limit
            if limit and processed_count + len(images) > limit:
                remaining = limit - processed_count
                images = images[:remaining]
                paths = paths[:remaining]

            # If 'image' is a list of dicts (HF format), extract bytes
            if images and isinstance(images[0], dict) and 'bytes' in images[0]:
                images = [img['bytes'] for img in images]

            tasks = [(img, path, output_dir) for img, path in zip(images, paths)]

            # Execute tasks
            results = list(executor.map(save_image, tasks))
            processed_count += sum(1 for r in results if r)

    print(f"Successfully converted {processed_count} images to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert Parquet dataset to image files")
    parser.add_argument("--input", type=str, required=True, help="Path to input Parquet file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save images")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of images to process")

    args = parser.parse_args()

    convert_parquet_to_images(args.input, args.output, args.workers, args.limit)

if __name__ == "__main__":
    main()
