import torch
from collections import OrderedDict
from argparse import ArgumentParser

import mmcv
from mmocr.models import build_detector  # noqa: F401

parser = ArgumentParser()
parser.add_argument('--config', help='Config file.', default='./seg_r31_1by16_fpnocr_academic.py')
parser.add_argument('--checkpoint', help='Checkpoint file.', default='./seg_r31_1by16_fpnocr_academic-72235b11.pth')
parser.add_argument('--out_file', help='Path to save visualized image.',default="./segocr.pt")
parser.add_argument(
    '--device', default='cuda:0', help='Device used for inference.')
args = parser.parse_args()

if isinstance(args.config, str):
    config = mmcv.Config.fromfile(args.config)

elif not isinstance(args.config, mmcv.Config):
    raise TypeError('config must be a filename or Config object, '
                    f'but got {type(args.config)}')

config.model.pretrained = None
config.model.train_cfg = None
model = build_detector(config.model, test_cfg=config.get('test_cfg'))
state_dict = torch.load(
    args.checkpoint, map_location=lambda storage, loc: storage)
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    new_state_dict.update(v)

del [new_state_dict['CLASSES']]

model.load_state_dict(new_state_dict)
model.eval()


model.to(args.device)

if config.data.test['type'] == 'ConcatDataset':
    config.data.test.pipeline = config.data.test['datasets'][
        0].pipeline

data = torch.rand(1, 3, 64, 192).cuda()
with torch.no_grad():
    traced_script_module = torch.jit.trace(model, data)
    traced_script_module.save(args.out_file)
