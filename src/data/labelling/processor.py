import asyncio
from datasets import load_dataset
from tqdm import tqdm
import os
import numpy as np
import cv2
from src.data.labelling.draw_boxes import draw_boxes
from src.data.labelling.agent import generate


# Load prompts from files
def read_prompt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# Path relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(SCRIPT_DIR, "prompt")
figure_prompt_content = read_prompt(os.path.join(PROMPT_DIR, "figure.md"))
text_prompt_content = read_prompt(os.path.join(PROMPT_DIR, "text.md"))

async def process_dataset(dataset):
    """
    Iterates through each sample and parses YOLO boxes.
    Only returns objects with class 3 (chart) and 14 (image).
    """
    processed_results = []
    target_classes = {3, 14}

    print("Processing dataset...")
    # Limiting to 5 samples for testing
    for idx, example in enumerate(tqdm(dataset['train'])):
        if idx >= 1: # Limit to 5 for test
            break
        img = example['image']
        W, H = img.size
        raw_labels = example['label_raw'].strip()

        sample_boxes = []
        if raw_labels:
            for line in raw_labels.split('\n'):
                parts = line.split()
                if not parts:
                    continue

                cls_id = int(parts[0])
                if cls_id in target_classes:
                    coords = list(map(float, parts[1:]))
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

        # Decide which image and prompt to use
        if sample_boxes:
            boxes_only = [obj['bbox'] for obj in sample_boxes]
            out_img, cropped_objects = draw_boxes(img_cv2, boxes_only)
            target_img = out_img
            prompt = figure_prompt_content
        else:
            target_img = img_cv2
            prompt = text_prompt_content
            cropped_objects = {}

        # Call the agent on the full image
        _, buffer = cv2.imencode('.jpg', target_img)
        image_bytes = buffer.tobytes()

        ocr_text = await generate(image_bytes, prompt=prompt)
        print(f"\n--- Sample {idx} OCR Result ---\n{ocr_text}\n")

        output_dir = "output_dev/draw_boxes"
        save_path = os.path.join(output_dir, f"sample_{idx}.png")
        cv2.imwrite(save_path, target_img)

        processed_results.append({
            'sample_idx': idx,
            'image': img,
            'objects': sample_boxes,
            'ocr_results': ocr_text,
            'crops': list(cropped_objects.keys())
        })

    return processed_results

if __name__ == "__main__":
    dataset_name = "daominhwysi/toanmath.com_25k"
    dataset = load_dataset(dataset_name)
    output_dir = "output_dev/draw_boxes"
    os.makedirs(output_dir, exist_ok=True)
    results = asyncio.run(process_dataset(dataset))
    print(f"\nProcessing complete. Found {len(results)} samples.")
