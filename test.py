import json

from mmocr.utils.fileio import list_to_file

with open('tests/data/ocr_toy_dataset/label.txt', 'r', encoding='utf-8') as f:
    annos = f.readlines()
labels = []
for anno in annos:
    img_name, text = anno.strip('\n').split(' ')
    labels.append(json.dumps({'filename': img_name, 'text': text}))
list_to_file('test.jsonl', labels)
