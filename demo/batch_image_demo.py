from argparse import ArgumentParser
from pathlib import Path

import mmcv

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument('save_path', help='Folder to save visualized images.')
    parser.add_argument('--images', nargs='+')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    # test multiple images
    results = model_inference(model, args.images)
    print(f'results: {results}')

    save_path = Path(args.save_path)
    for img_path, result in zip(args.images, results):

        out_file = save_path / f'result_{Path(img_path).stem}.png'

        # show the results
        img = model.show_result(
            img_path, result, out_file=str(out_file), show=False)
        if args.imshow:
            mmcv.imshow(img, f'predicted results ({img_path})')


if __name__ == '__main__':
    main()
