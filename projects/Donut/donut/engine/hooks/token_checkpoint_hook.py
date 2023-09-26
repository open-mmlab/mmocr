import torch
from mmocr.registry import HOOKS
from mmengine.model import MMDistributedDataParallel
from mmengine.hooks import CheckpointHook


@HOOKS.register_module()
class TokenCheckpointHook(CheckpointHook):
    """
    """

    def before_train(self, runner):
        """
        """
        super().before_train(runner=runner)
        if isinstance(runner.model, MMDistributedDataParallel):
            runner.model.module.decoder.tokenizer.save_vocabulary(self.out_dir)
        else:
            runner.model.decoder.tokenizer.save_vocabulary(self.out_dir)
