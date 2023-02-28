# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch

prefix_mapping = {
    'backbone.0.body': 'backbone',
    'input_proj': 'encoder.input_proj',
    'transformer': 'decoder',
    'vocab_embed.layers.': 'decoder.vocab_embed.layer-'
}


def adapt(model_path, save_path):
    model = torch.load(model_path)
    model_dict = model['model']
    new_model_dict = model_dict.copy()

    for k, v in model_dict.items():
        for old_prefix, new_prefix in prefix_mapping.items():
            if k.startswith(old_prefix):
                new_k = k.replace(old_prefix, new_prefix)
                new_model_dict[new_k] = v
                del new_model_dict[k]
                break
    model['state_dict'] = new_model_dict
    del model['model']
    torch.save(model, save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Adapt the pretrained checkpoints from SPTS official '
        'implementation.')
    parser.add_argument(
        'model_path', type=str, help='Path to the source model')
    parser.add_argument(
        'out_path', type=str, help='Path to the converted model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    adapt(args.model_path, args.out_path)
