import argparse
import os
from collections import OrderedDict

import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--save-dir', default='data')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)

    # get model using:
    # git clone --branch official \
    #     https://huggingface.co/naver-clova-ix/donut-base
    assert os.path.exists(args.model), args.model
    assert args.model[-4:] == '.bin', 'the model name is pytorch_model.bin'
    model_state_dict = torch.load(args.model)

    # extract weights
    encoder_state_dict = OrderedDict()
    decoder_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if k.startswith('encoder.'):
            new_k = k[len('encoder.'):]
            encoder_state_dict[new_k] = v
        elif k.startswith('decoder.'):
            new_k = k[len('decoder.'):]
            decoder_state_dict[new_k] = v

    # save weights
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    torch.save(encoder_state_dict,
               os.path.join(args.save_dir, 'donut_base_encoder.pth'))
    torch.save(decoder_state_dict,
               os.path.join(args.save_dir, 'donut_base_decoder.pth'))


if __name__ == '__main__':
    main()
