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

print(dataset['train'][0].keys())
#{'image': <PIL.WebPImagePlugin.WebPImageFile image mode=RGB size=2481x3508 at 0x7175BACD6FB0>, 'file_name': 'bai-giang-toan-10-chu-de-menh-de-va-tap-hop-le-quang-xe_page_35.webp', 'label_raw': '12 0.892382 0.040194 0.083031 0.015393\n17 0.134220 0.093501 0.076582 0.019384\n22 0.506247 0.130844 0.844015 0.038769\n22 0.509270 0.176026 0.110036 0.020810\n17 0.133817 0.272520 0.075776 0.019384\n22 0.465740 0.316420 0.764611 0.052452\n22 0.509674 0.368301 0.108424 0.021095\n17 0.133615 0.464652 0.076985 0.019954\n22 0.509875 0.560576 0.109633 0.021380\n17 0.134220 0.657212 0.076582 0.019099\n22 0.505643 0.695410 0.843611 0.040194\n22 0.509472 0.740023 0.108827 0.021095\n17 0.139258 0.836374 0.088271 0.019954\n22 0.505844 0.871864 0.845627 0.037343\n22 0.408505 0.904219 0.606610 0.020525\n22 0.491334 0.931157 0.772269 0.021950\n8 0.197501 0.962514 0.234583 0.015108'}
