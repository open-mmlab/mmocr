import json
import os

from mmengine.hooks import CheckpointHook
from mmengine.model import MMDistributedDataParallel

from mmocr.registry import HOOKS


@HOOKS.register_module()
class TokenCheckpointHook(CheckpointHook):
    """"""

    def before_train(self, runner):
        """save tokenizer."""
        super().before_train(runner=runner)
        if isinstance(runner.model, MMDistributedDataParallel):
            tokenizer = runner.model.module.decoder.tokenizer
            tokenizer.save_vocabulary(self.out_dir)
            added_vocab = tokenizer.get_added_vocab()
            with open(os.path.join(self.out_dir, 'added_tokens.json'),
                      'w') as f:
                json.dump(added_vocab, f)
        else:
            tokenizer = runner.model.decoder.tokenizer
            tokenizer.save_vocabulary(self.out_dir)
            added_vocab = tokenizer.get_added_vocab()
            with open(os.path.join(self.out_dir, 'added_tokens.json'),
                      'w') as f:
                json.dump(added_vocab, f)
