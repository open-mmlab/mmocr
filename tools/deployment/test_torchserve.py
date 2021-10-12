from argparse import ArgumentParser

import numpy as np
import requests

from mmocr.apis import init_detector, model_inference


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('model_name', help='The model name in the server')
    parser.add_argument(
        '--inference-addr',
        default='127.0.0.1:8080',
        help='Address and port of the inference server')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    model_results = model_inference(model, args.img)
    model.show_results(
        args.img,
        model_results,
        title='model_results',
        score_thr=args.score_thr)
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    with open(args.img, 'rb') as image:
        response = requests.post(url, image)
    serve_results = response.json()
    model.show_results(
        args.img,
        serve_results,
        title='serve_results',
        score_thr=args.score_thr)
    assert serve_results.keys() == model_results.keys()
    for key in serve_results.keys():
        if isinstance(serve_results[0], (int, float)):
            assert np.allclose(model_results[key], serve_results[key])
        if isinstance(serve_results[0], str):
            assert model_results[key] == serve_results[key]


if __name__ == '__main__':
    args = parse_args()
    main(args)
