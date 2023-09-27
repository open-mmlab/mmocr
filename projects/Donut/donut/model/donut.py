import random
import re
from typing import Any, Union

import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData
from torch.nn.utils.rnn import pad_sequence
from transformers.file_utils import ModelOutput

from mmocr.registry import MODELS
from ..datasets.cord_dataset import SPECIAL_TOKENS


@MODELS.register_module()
class Donut(BaseModel):

    def __init__(self,
                 data_preprocessor=None,
                 encoder=dict(
                     type='SwinEncoder',
                     input_size=[1280, 960],
                     align_long_axis=False,
                     window_size=10,
                     encoder_layer=[2, 2, 14, 2],
                     name_or_path=''),
                 decoder=dict(
                     type='BARTDecoder',
                     max_position_embeddings=None,
                     task_start_token='<s>',
                     prompt_end_token=None,
                     decoder_layer=4,
                     name_or_path=''),
                 max_length=768,
                 ignore_mismatched_sizes=True,
                 sort_json_key: bool = True,
                 ignore_id: int = -100,
                 init_cfg=dict()):
        super().__init__(data_preprocessor, init_cfg)

        self.max_length = max_length
        self.ignore_mismatched_sizes = ignore_mismatched_sizes
        self.sort_json_key = sort_json_key
        self.ignore_id = ignore_id

        self.encoder = MODELS.build(encoder)

        decoder['max_position_embeddings'] = max_length if decoder[
            'max_position_embeddings'] is None else decoder[
                'max_position_embeddings']
        self.decoder = MODELS.build(decoder)

    def init_weights(self):
        super().init_weights()
        self.decoder.add_special_tokens(SPECIAL_TOKENS)
        self.decoder.add_special_tokens(
            [self.decoder.task_start_token, self.decoder.prompt_end_token])
        return

    def get_input_ids_val(self, data_samples):
        # input_ids
        decoder_input_ids = list()
        batch_prompt_end_index = list()
        batch_processed_parse = list()

        for sample in data_samples:
            if hasattr(sample, 'parses_json'):
                assert isinstance(sample.parses_json, list)
                gt_jsons = sample.parses_json
            else:
                print(sample.keys())
                raise KeyError

            # load json from list of json
            gt_token_sequences = []
            for gt_json in gt_jsons:
                gt_token = self.json2token(
                    gt_json,
                    update_special_tokens_for_json_key=False,
                    sort_json_key=self.sort_json_key)
                gt_token_sequences.append(self.decoder.task_start_token +
                                          gt_token +
                                          self.decoder.tokenizer.eos_token)
            # can be more than one, e.g., DocVQA Task 1
            token_index = random.randint(0, len(gt_token_sequences) - 1)
            processed_parse = gt_token_sequences[token_index]

            input_ids = self.decoder.tokenizer(
                processed_parse,
                add_special_tokens=False,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )['input_ids'].squeeze(0)

            # return prompt end index instead of target output labels
            prompt_end_index = torch.nonzero(
                input_ids == self.decoder.prompt_end_token_id).sum()
            batch_prompt_end_index.append(prompt_end_index)
            batch_processed_parse.append(processed_parse)

            decoder_input_ids.append(input_ids[:-1])

            sample.gt_instances['parses_json'] = [gt_jsons[token_index]
                                                  ]  # [] for len check
            sample.gt_instances['parses'] = [processed_parse]

        decoder_input_ids = torch.stack(decoder_input_ids, dim=0)
        return decoder_input_ids, batch_prompt_end_index, batch_processed_parse

    def get_input_ids_train(self, data_samples):
        # input_ids
        decoder_input_ids = list()
        decoder_labels = list()

        for sample in data_samples:
            assert isinstance(sample.parses_json, list)
            gt_jsons = sample.parses_json

            # load json from list of json
            gt_token_sequences = []
            for gt_json in gt_jsons:
                gt_token = self.json2token(
                    gt_json,
                    update_special_tokens_for_json_key=False,
                    sort_json_key=self.sort_json_key)
                gt_token_sequences.append(self.decoder.task_start_token +
                                          gt_token +
                                          self.decoder.tokenizer.eos_token)
            # can be more than one, e.g., DocVQA Task 1
            token_index = random.randint(0, len(gt_token_sequences) - 1)
            processed_parse = gt_token_sequences[token_index]

            input_ids = self.decoder.tokenizer(
                processed_parse,
                add_special_tokens=False,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )['input_ids'].squeeze(0)

            labels = input_ids.clone()
            # model doesn't need to predict pad token
            labels[labels ==
                   self.decoder.tokenizer.pad_token_id] = self.ignore_id
            # model doesn't need to predict prompt (for VQA)
            labels[:torch.nonzero(
                labels == self.decoder.prompt_end_token_id).sum() +
                   1] = self.ignore_id
            decoder_labels.append(labels[1:])

            decoder_input_ids.append(input_ids[:-1])
            sample.gt_instances['parses_json'] = [gt_jsons[token_index]]
            sample.gt_instances['parses'] = [processed_parse]

        decoder_input_ids = torch.stack(decoder_input_ids, dim=0)
        decoder_labels = torch.stack(decoder_labels, dim=0)
        return decoder_input_ids, decoder_labels

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='test')  # type: ignore

    def forward(self,
                inputs: torch.Tensor,
                data_samples=None,
                mode: str = 'tensor',
                **kwargs):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'test':
            return self.test(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(self, inputs, data_samples=None):
        """Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner.

        Args:
            inputs: (batch_size, num_channels, height, width)
            decoder_input_ids: (batch_size, sequence_length, embedding_dim)
            decode_labels: (batch_size, sequence_length)
        """
        encoder_outputs = self.encoder(inputs)

        input_ids, labels = self.get_input_ids_train(data_samples)
        input_ids = input_ids.to(encoder_outputs.device)
        labels = labels.to(encoder_outputs.device)

        decoder_outputs = self.decoder.loss(
            input_ids=input_ids,
            encoder_hidden_states=encoder_outputs,
            labels=labels,
            data_samples=data_samples)
        return decoder_outputs

    def predict(self, inputs, data_samples=None):
        """Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner.

        Args:
            inputs: (batch_size, num_channels, height, width)
            decoder_input_ids: (batch_size, sequence_length, embedding_dim)
            decode_labels: (batch_size, sequence_length)
        """
        encoder_outputs = self.encoder(inputs)

        decoder_input_ids, prompt_end_idxs, answers = self.get_input_ids_val(
            data_samples)

        prompt_tensors = pad_sequence(
            [
                input_id[:end_idx + 1] for input_id, end_idx in zip(
                    decoder_input_ids, prompt_end_idxs)
            ],
            batch_first=True,
        )
        decoder_input_ids = decoder_input_ids.to(encoder_outputs.device)

        if len(encoder_outputs.size()) == 1:
            encoder_outputs = encoder_outputs.unsqueeze(0)

        return_attentions = False

        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)
        prompt_tensors = prompt_tensors.to(encoder_outputs.device)

        # get decoder output
        encoder_outputs = ModelOutput(
            last_hidden_state=encoder_outputs, attentions=None)

        decoder_output = self.decoder.model.generate(
            decoder_input_ids=prompt_tensors,
            encoder_outputs=encoder_outputs,
            max_length=self.max_length,
            early_stopping=True,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.decoder.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=return_attentions,
        )

        output = {'predictions': list(), 'predictions_json': list()}
        for i, seq in enumerate(
                self.decoder.tokenizer.batch_decode(decoder_output.sequences)):
            seq = seq.replace(self.decoder.tokenizer.eos_token, '')
            seq = seq.replace(self.decoder.tokenizer.pad_token, '')
            # remove first task start token
            seq = re.sub(r'<.*?>', '', seq, count=1).strip()
            output['predictions'].append(seq)
            output['predictions_json'].append(self.token2json(seq))

            answer = answers[i]
            answer = re.sub(r'<.*?>', '', answer, count=1)
            answer = answer.replace(self.decoder.tokenizer.eos_token, '')

            data_samples[i].pred_instances = InstanceData(
                parses=[seq], parses_json=[self.token2json(seq)])
            data_samples[i].gt_instances['parses'] = [answer]

        if return_attentions:
            output['attentions'] = {
                'self_attentions': decoder_output.decoder_attentions,
                'cross_attentions': decoder_output.cross_attentions,
            }

        return data_samples

    def test(self, inputs, data_samples=None):
        """Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner.

        Args:
            inputs: (batch_size, num_channels, height, width)
            decoder_input_ids: (batch_size, sequence_length, embedding_dim)
            decode_labels: (batch_size, sequence_length)
        """
        encoder_outputs = self.encoder(inputs)
        if len(encoder_outputs.size()) == 1:
            encoder_outputs = encoder_outputs.unsqueeze(0)
        encoder_outputs = ModelOutput(
            last_hidden_state=encoder_outputs, attentions=None)

        prompt_tensors = self.decoder.tokenizer(
            self.decoder.task_start_token,
            add_special_tokens=False,
            return_tensors='pt')['input_ids']
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)
        prompt_tensors = prompt_tensors.to(inputs.device)

        return_attentions = False
        decoder_output = self.decoder.model.generate(
            decoder_input_ids=prompt_tensors,
            encoder_outputs=encoder_outputs,
            max_length=self.max_length,
            early_stopping=True,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.decoder.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=return_attentions,
        )

        output = {'predictions': list(), 'predictions_json': list()}
        for i, seq in enumerate(
                self.decoder.tokenizer.batch_decode(decoder_output.sequences)):
            seq = seq.replace(self.decoder.tokenizer.eos_token, '')
            seq = seq.replace(self.decoder.tokenizer.pad_token, '')
            # remove first task start token
            seq = re.sub(r'<.*?>', '', seq, count=1).strip()
            output['predictions'].append(seq)
            output['predictions_json'].append(self.token2json(seq))

            answer = data_samples[i].parses_json
            data_samples[i].pred_instances = InstanceData(
                parses=[seq], parses_json=[self.token2json(seq)])
            data_samples[i].gt_instances['parses_json'] = answer

        if return_attentions:
            output['attentions'] = {
                'self_attentions': decoder_output.decoder_attentions,
                'cross_attentions': decoder_output.cross_attentions,
            }

        return data_samples

    def json2token(self,
                   obj: Any,
                   update_special_tokens_for_json_key: bool = True,
                   sort_json_key: bool = True):
        """Convert an ordered JSON object into a token sequence."""
        if type(obj) == dict:
            if len(obj) == 1 and 'text_sequence' in obj:
                return obj['text_sequence']
            else:
                output = ''
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.decoder.add_special_tokens(
                            [fr'<s_{k}>', fr'</s_{k}>'])
                    output += (fr'<s_{k}>' + self.json2token(
                        obj[k], update_special_tokens_for_json_key,
                        sort_json_key) + fr'</s_{k}>')
                return output
        elif type(obj) == list:
            return r'<sep/>'.join([
                self.json2token(item, update_special_tokens_for_json_key,
                                sort_json_key) for item in obj
            ])
        else:
            obj = str(obj)
            if f'<{obj}/>' in self.decoder.tokenizer.all_special_tokens:
                obj = f'<{obj}/>'  # for categorical special tokens
            return obj

    def token2json(self, tokens, is_inner_value=False):
        """Convert a (generated) token seuqnce into an ordered JSON format."""
        output = dict()

        while tokens:
            start_token = re.search(r'<s_(.*?)>', tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(fr'</s_{key}>', tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, '')
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f'{start_token_escaped}(.*?){end_token_escaped}', tokens,
                    re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    # non-leaf node
                    if r'<s_' in content and r'</s_' in content:
                        value = self.token2json(content, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r'<sep/>'):
                            leaf = leaf.strip()
                            if (leaf in
                                    self.decoder.tokenizer.get_added_vocab()
                                    and leaf[0] == '<' and leaf[-2:] == '/>'):
                                leaf = leaf[
                                    1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) +
                                len(end_token):].strip()
                if tokens[:6] == r'<sep/>':  # non-leaf nodes
                    return [output] + self.token2json(
                        tokens[6:], is_inner_value=True)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {'text_sequence': tokens}
