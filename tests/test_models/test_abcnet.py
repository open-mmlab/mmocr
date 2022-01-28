from mmcv import Config

from mmocr.models import build_detector


def test_abcnet(cfg):
    cfg = Config.fromfile(cfg)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.backbone.state_dict()


if __name__ == '__main__':
    cfg = 'configs/textdet/abcnet/abcnet.py'
    test_abcnet(cfg)
