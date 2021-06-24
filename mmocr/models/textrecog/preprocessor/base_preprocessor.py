from mmcv.runner import BaseModule

from mmocr.models.builder import PREPROCESSOR


@PREPROCESSOR.register_module()
class BasePreprocessor(BaseModule):
    """Base Preprocessor class for text recognition."""
    '''
    def init_weights(self):
        pass
    '''

    def forward(self, x, **kwargs):
        return x
