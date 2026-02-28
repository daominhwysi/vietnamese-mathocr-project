from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

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


# Initialize the counter
class_counts = Counter()

print("Counting class distributions...")

# Iterate through the training set
for example in tqdm(dataset['train']):
    raw_labels = example['label_raw'].strip()
    if not raw_labels:
        continue

    # YOLO format: each line starts with the class ID
    for line in raw_labels.split('\n'):
        parts = line.split()
        if parts:
            class_id = int(parts[0])
            class_counts[class_id] += 1

# Display results
print("\n" + "="*45)
print(f"{'ID':<5} | {'Label Name':<20} | {'Count':<10}")
print("-" * 45)

total_boxes = sum(class_counts.values())

# Sort by ID or Count (change to class_counts.most_common() to sort by count)
for class_id in sorted(class_counts.keys()):
    label_name = id2label.get(class_id, "Unknown")
    count = class_counts[class_id]
    percentage = (count / total_boxes) * 100
    print(f"{class_id:<5} | {label_name:<20} | {count:<10} ({percentage:>5.2f}%)")

print("="*45)
print(f"Total Bounding Boxes: {total_boxes}")
