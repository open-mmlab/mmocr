from argparse import ArgumentParser

import mmcv

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file.')
    parser.add_argument('config', help='Config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument('out_file', help='Path to save visualized image.')
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

    # test a single image
    result = model_inference(model, args.img)
    print(f'result: {result}')

    # show the results
    img = model.show_result(
        args.img, result, out_file=args.out_file, show=False)

    if img is None:
        img = mmcv.imread(args.img)

    mmcv.imwrite(img, args.out_file)
    if args.imshow:
        mmcv.imshow(img, 'predicted results')


if __name__ == '__main__':
    main()
