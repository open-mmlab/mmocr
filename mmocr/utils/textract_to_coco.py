from email.mime import image
from nuocr import batch_ocr
import glob
from tqdm import tqdm
import imagesize
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('imagesize')
DEFAULT_CATEGORY = 1


def ocr(img):
    res = batch_ocr([img], engine='textract', line_flag=True)
    return res[0]['annotations']

def _image_dict(img, id):
        width, height = imagesize.get(img)
        path_name = "images/" + img.split('/')[-1]
        return {
            "id": id,
            "width": width,
            "height": height,
            "file_name": path_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }

def _annotation_dict(annotation, img_id, ann_id):
        y1, y2, x1, x2 = annotation['rect']
        return {
            "id": ann_id,
            "image_id": img_id,
            "category_id": DEFAULT_CATEGORY,
            "segmentation": [],
            "area": (x2 - x1) * (y2 - y1),
            "bbox": [
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ],
            "iscrowd": 0,
            "attributes": {
                "text": annotation['text'],
                "conf": annotation['conf']}
        }

def get_coco_dict():
    return {
            "licenses": [
                {
                    "name": "",
                    "id": 0,
                    "url": ""
                }
            ],
            "info": 
                {
                    "contributor": "",
                    "date_created": "",
                    "description": "",
                    "url": "",
                    "version": "",
                    "year": ""
                },
            "categories": [
                {
                    "id": 1,
                    "name": "other",
                    "supercategory": ""
                }
            ],
            "images": [],
            "annotations": []
        }

def get_annotations(imgs):
    coco = get_coco_dict()
    count_img_id = 1
    count_ann_id = 1
    for img in tqdm(imgs):
        res = ocr(img)
        img_dict = _image_dict(img, count_img_id)
        coco['images'].append(img_dict)
        for annotation in res:
            ann_dict = _annotation_dict(annotation, count_img_id, count_ann_id)
            ann_dict['image_id'] = img_dict['id']
            coco['annotations'].append(ann_dict)
            count_ann_id += 1
        count_img_id += 1
    
    return coco

def coco_to_json(annotations):
    import json
    with open('instances_default.json', 'w') as f:
        json.dump(annotations, f)

def main(image_path):
    
    imgs = glob.glob(image_path)
    coco_ann = get_annotations(imgs)
    coco_to_json(coco_ann)


if __name__ == '__main__':
    image_path = 'images/*'
    main(image_path)