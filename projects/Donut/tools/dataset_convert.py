import os
import json
import argparse
from datasets import load_dataset
from io import BytesIO
from PIL import Image
import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='naver-clova-ix/cord-v2')
    parser.add_argument('--save-dir', default='datasets/cord-v2')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)

    data_dir = args.data_dir
    data_save_dir = args.save_dir

    dataset = load_dataset(data_dir, split=None)

    img_id = 0
    for split in dataset.keys():
        split_save_dir = os.path.join(data_save_dir, split)
        split_image_dir = os.path.join(split_save_dir, 'images')
        if not os.path.exists(split_save_dir):
            os.makedirs(split_save_dir)
            os.makedirs(split_image_dir, exist_ok=True)
        split_meta_save_path = os.path.join(split_save_dir, 'metadata.jsonl')

        metadata = []
        for sample in tqdm.tqdm(dataset[split]):
            image = sample['image']
            if isinstance(image, dict):
                image = Image.open(BytesIO(image['bytes']))
            image.save(os.path.join(split_image_dir, f'{img_id}.jpg'))
            image_name = f'images/{img_id}.jpg'
            ground_truth = sample['ground_truth']
            metadata.append(json.dumps({'file_name': image_name, 'ground_truth': ground_truth}, ensure_ascii=False))
            img_id += 1

        with open(split_meta_save_path, 'w') as f:
            f.write('\n'.join(metadata))


if __name__ == '__main__':
    main()
