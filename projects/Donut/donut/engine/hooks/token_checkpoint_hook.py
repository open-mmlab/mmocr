import os
import json
import torch
from mmocr.registry import HOOKS
from mmengine.model import MMDistributedDataParallel
from mmengine.hooks import CheckpointHook


@HOOKS.register_module()
class TokenCheckpointHook(CheckpointHook):
    """
    """

    def before_train(self, runner):
        """save tokenizer
        """
        super().before_train(runner=runner)
        if isinstance(runner.model, MMDistributedDataParallel):
            runner.model.module.decoder.tokenizer.save_vocabulary(self.out_dir)
            added_vocab = runner.model.module.decoder.tokenizer.get_added_vocab()
            with open(os.path.join(self.out_dir, 'added_tokens.json'), 'w') as f:
                json.dump(added_vocab, f)
        else:
            runner.model.decoder.tokenizer.save_vocabulary(self.out_dir)
            added_vocab = runner.model.decoder.tokenizer.get_added_vocab()
            with open(os.path.join(self.out_dir, 'added_tokens.json'), 'w') as f:
                json.dump(added_vocab, f)
