from argparse import ArgumentParser

from mmdet.apis import init_detector
from mmocr.apis.inference import text_model_inference

from mmocr.datasets import build_dataset  # NOQA
from mmocr.models import build_detector  # NOQA


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    # test a single text
    input_sentence = input('Please enter a sentence you want to test: ')
    result = text_model_inference(model, input_sentence)

    # show the results
    for i in range(len(result)):
        pred_entities = result[i]
        text = input_sentence
        for entity in pred_entities:
            if entity[2] > entity[1] and entity[1] < len(text):
                if entity[2] > len(text):
                    entity[2] = len(text)
                print(f'{entity[0]}: {text[entity[1]:entity[2] + 1]}')


if __name__ == '__main__':
    main()
