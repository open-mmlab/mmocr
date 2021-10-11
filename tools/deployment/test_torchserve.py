from argparse import ArgumentParser

# import numpy as np
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


def parse_result(inputs):
    bbox = []
    # text = []
    # score = []
    for input in inputs:
        if input.get('bbox', None):
            bbox.append(input)
    return bbox


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    model_results = model_inference(model, args.img)
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    with open(args.img, 'rb') as image:
        response = requests.post(url, image)
    server_results = parse_result(response.json())
    return model_results, server_results

    # assert np.allclose(model_result[i], server_result[i])


if __name__ == '__main__':
    args = parse_args()
    main(args)
