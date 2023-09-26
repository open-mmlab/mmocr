from collections import OrderedDict
import torch


def main():
    # load model using git clone --branch official https://huggingface.co/naver-clova-ix/donut-base
    model_state_dict = torch.load('donut-base/pytorch_model.bin')


    # extract weights
    encoder_state_dict = OrderedDict()
    decoder_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if k.startswith('encoder.'):
            new_k = k[len('encoder.'):]
            encoder_state_dict[new_k] = v
        elif k.startswith('decoder.'):
            if k.startswith('decoder.model.'):
                new_k = k[len('decoder.model.'):]
            else:
                new_k = k[len('decoder.'):]
            decoder_state_dict[new_k] = v

    # save weights
    torch.save(decoder_state_dict, 'data/donut_cord_decoder.pth')
    torch.save(decoder_state_dict, 'data/donut_cord_decoder.pth')


if __name__ == '__main__':
    main()
