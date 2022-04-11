import json

def _get_mmocr_dict(img):
    return {
        "file_name": img['file_name'],
        "height": img['height'],
        "width": img['width'],
        "annotations": []
    }

def convert(path_to_coco, val_indices = []):
    '''
    Convert COCO json to mmocr txt. 
    Can be separate validation and train .txt files.
    
    Args:
        path_to_coco: str
            Path to coco json
        val_indices: str
            Indices of images to be used for validation

    '''
    val = []
    train = []
    with open(path_to_coco) as json_file:
        data = json.load(json_file)

    for idx, img in enumerate(data['images']):
        mmocr_dict = _get_mmocr_dict(img)

        for ann in data['annotations']:
            if ann['image_id'] == img['id']:
                x1, y1, w, h = ann['bbox']
                x2 = x1 + w
                y2 = y1
                x3 = x2
                y3 = y1 + h
                x4 = x1
                y4 = y3
                mmocr_box = {
                    "box": [x1, y1, x2, y2, x3, y3, x4, y4],
                    "text": ann['attributes']['text'],
                    "label": ann['category_id'] - 1,
                }

                mmocr_dict['annotations'].append(mmocr_box)

        if idx in val_indices:
            val.append(json.dumps(mmocr_dict))
        else:
            train.append(json.dumps(mmocr_dict))
            
    with open('train.txt', 'w') as f:
        for line in train:
            f.write(line+"\n")

    with open('val.txt', 'w') as f:
        for line in val:
            f.write(line+"\n")

def main():
    convert('instances_default.json', [])

if __name__ == '__main__':
    main()
    
    