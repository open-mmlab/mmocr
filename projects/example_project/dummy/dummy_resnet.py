from mmdet.models.backbones import ResNet as MMDET_RESNET

from mmocr.registry import MODELS


@MODELS.register_module()
class DummyResNet(MMDET_RESNET):
    """Implements a dummy ResNet wrapper for demonstration purpose.

    Args:
        **kwargs: All the arguments are passed to the parent class.
    """

    def __init__(self, **kwargs) -> None:
        print('Hello world!')
        super().__init__(**kwargs)
