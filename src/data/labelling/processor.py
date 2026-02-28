from datasets import load_dataset
from tqdm import tqdm
import os
import numpy as np
import cv2
from src.data.labelling.draw_boxes import draw_boxes



def process_dataset(dataset):
    """
    Iterates through each sample and parses YOLO boxes.
    Only returns objects with class 5 (formula) and 14 (image).
    """
    processed_results = []
    target_classes = {3, 14}

    print("Processing dataset...")
    for idx, example in enumerate(tqdm(dataset['train'])):
        img = example['image']
        W, H = img.size
        raw_labels = example['label_raw'].strip()

        if not raw_labels:
            continue

        sample_boxes = []
        for line in raw_labels.split('\n'):
            parts = line.split()
            if not parts:
                continue

            cls_id = int(parts[0])
            if cls_id in target_classes:
                # YOLO format: x_center, y_center, width, height (normalized)
                coords = list(map(float, parts[1:]))

                # Convert to pixel coordinates (x1, y1, x2, y2) if needed,
                # but following the prompt "parse yolo boxes" we keep them as is
                # or store in a structured way.
                x_c, y_c, w_n, h_n = coords

                x1 = (x_c - w_n / 2) * W
                y1 = (y_c - h_n / 2) * H
                x2 = (x_c + w_n / 2) * W
                y2 = (y_c + h_n / 2) * H

                sample_boxes.append({
                    'bbox': [x1, y1, x2, y2],
                    'cls_id': cls_id
                })
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        boxes = [obj['bbox'] for obj in sample_boxes]

        out_img, _ = draw_boxes(
            img_cv2,
            boxes
        )
        output_dir = "output_dev/draw_boxes"
        save_path = os.path.join(output_dir, f"sample_{idx}.png")
        cv2.imwrite(save_path, out_img)
        if sample_boxes:
            processed_results.append({
                'sample_idx': idx,
                'image': img,
                'objects': sample_boxes
            })
        #halt for testing
        if idx == 10:
            break
    return processed_results

if __name__ == "__main__":
  dataset_name = "daominhwysi/toanmath.com_25k"
  dataset = load_dataset(dataset_name)
  output_dir = "output_dev/draw_boxes"
  os.makedirs(output_dir, exist_ok=True)
  results = process_dataset(dataset)
  print(f"\nProcessing complete. Found {len(results)} samples with target classes.")
