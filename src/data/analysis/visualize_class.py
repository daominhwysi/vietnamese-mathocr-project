import os
from tqdm import tqdm
from PIL import Image, ImageDraw
from collections import Counter
from datasets import load_dataset
dataset_name = "daominhwysi/toanmath.com_25k"
dataset = load_dataset(dataset_name)

id2label = {0: 'abstract',
 1: 'algorithm',
 2: 'aside_text',
 3: 'chart',
 4: 'content',
 5: 'formula',
 6: 'doc_title',
 7: 'figure_title',
 8: 'footer',
 9: 'footer',
 10: 'footnote',
 11: 'formula_number',
 12: 'header',
 13: 'header',
 14: 'image',
 15: 'formula',
 16: 'number',
 17: 'paragraph_title',
 18: 'reference',
 19: 'reference_content',
 20: 'seal',
 21: 'table',
 22: 'text',
 23: 'text',
 24: 'vision_footnote'}
# 1. Setup paths and limits
BASE_OUTPUT_DIR = "output_dev/visualized_cls_id"
SAMPLES_PER_CLASS = 100
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Tracks how many images we've saved for each class
saved_counts = Counter()

print("Starting visualization process...")

# 2. Iterate through the dataset once
for idx, example in enumerate(tqdm(dataset['train'])):
    img = example['image']
    W, H = img.size
    raw_labels = example['label_raw'].strip()

    if not raw_labels:
        continue

    # Parse all labels in this image
    # We group by class_id so we can draw one image per class found
    labels_by_class = {}
    for line in raw_labels.split('\n'):
        parts = line.split()
        if not parts: continue

        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        if cls_id not in labels_by_class:
            labels_by_class[cls_id] = []
        labels_by_class[cls_id].append(coords)

    # 3. For each class found in this image, check if we still need samples
    for cls_id, bboxes in labels_by_class.items():
        if saved_counts[cls_id] >= SAMPLES_PER_CLASS:
            continue

        # Create class directory if not exists
        class_dir = os.path.join(BASE_OUTPUT_DIR, str(cls_id))
        os.makedirs(class_dir, exist_ok=True)

        # Draw only the bboxes for THIS specific class
        # We work on a copy to keep the original clean for other classes in same image
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)

        for bbox in bboxes:
            x_c, y_c, w_n, h_n = bbox

            # Convert YOLO to Pixel coordinates
            left = (x_c - w_n / 2) * W
            top = (y_c - h_n / 2) * H
            right = (x_c + w_n / 2) * W
            bottom = (y_c + h_n / 2) * H

            # Draw thick red rectangle
            draw.rectangle([left, top, right, bottom], outline="red", width=3)

            # Optional: Add label name for easier debugging
            label_text = f"{cls_id}: {id2label.get(cls_id, 'Unknown')}"
            draw.text((left, top - 10), label_text, fill="red")

        # Save the image
        file_name = f"sample_{idx}.jpg"
        draw_img.save(os.path.join(class_dir, file_name))

        saved_counts[cls_id] += 1

    # 4. Stop early if all classes have reached 100 samples
    # (Note: Some classes may never reach 100 if they are rare in the dataset)
    if all(count >= SAMPLES_PER_CLASS for count in saved_counts.values()) and len(saved_counts) >= len(id2label):
        print("Reached 100 samples for all detected classes.")
        break

print("\nVisualization complete.")
print("Summary of saved samples:")
for cls_id in sorted(saved_counts.keys()):
    print(f"Class {cls_id} ({id2label.get(cls_id, 'Unknown')}): {saved_counts[cls_id]} samples")
