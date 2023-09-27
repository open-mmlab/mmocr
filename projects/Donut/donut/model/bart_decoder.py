from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from transformers import MBartConfig, MBartForCausalLM, XLMRobertaTokenizer
from transformers.file_utils import ModelOutput

from mmocr.registry import MODELS


@MODELS.register_module()
class BARTDecoder(BaseModel):
    """Donut Decoder based on Multilingual BART Set the initial weights and
    configuration with a pretrained multilingual BART model, and modify the
    detailed configurations as a Donut decoder.

    Args:
        decoder_layer:
            Number of layers of BARTDecoder
        max_position_embeddings:
            The maximum sequence length to be trained
    """

    def __init__(self,
                 decoder_layer: int,
                 max_position_embeddings: int,
                 task_start_token='<s>',
                 prompt_end_token=None,
                 tokenizer_cfg=dict(
                     type='XLMRobertaTokenizer',
                     checkpoint='hyunwoongko/asian-bart-ecjk'),
                 init_cfg=dict()):
        super().__init__(init_cfg=init_cfg)
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_position_embeddings
        self.task_start_token = task_start_token
        if prompt_end_token:
            self.prompt_end_token = prompt_end_token
        else:
            self.prompt_end_token = task_start_token

        self.tokenizer_cfg = tokenizer_cfg
        if tokenizer_cfg['type'] == 'XLMRobertaTokenizer' and tokenizer_cfg[
                'checkpoint']:
            self._tokenizer = XLMRobertaTokenizer.from_pretrained(
                self.tokenizer_cfg['checkpoint'])

        self.model = MBartForCausalLM(
            config=MBartConfig(
                is_decoder=True,
                is_encoder_decoder=False,
                add_cross_attention=True,
                decoder_layers=self.decoder_layer,
                max_position_embeddings=self.max_position_embeddings,
                vocab_size=len(self.tokenizer),
                scale_embedding=True,
                add_final_layer_norm=True,
            ))
        # to get cross attentions and utilize `generate` function
        self.model.forward = self.forward
        # to get cross-attention
        self.model.config.is_encoder_decoder = True
        # <sep/> is used for representing a list in a JSON
        self.add_special_tokens(['<sep/>'])
        pad_token_id = self.tokenizer.pad_token_id
        self.model.model.decoder.embed_tokens.padding_idx = pad_token_id
        prepare_inputs = self.prepare_inputs_for_inference
        self.model.prepare_inputs_for_generation = prepare_inputs

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def prompt_end_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def init_weights(self):
        super().init_weights()

        # weight init with asian-bart
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            bart_state_dict = MBartForCausalLM.from_pretrained(
                'hyunwoongko/asian-bart-ecjk').state_dict()
        else:
            bart_state_dict = OrderedDict()
            model_state_dict = torch.load(self.init_cfg['checkpoint'])
            for k, v in model_state_dict.items():
                if k.startswith('model.'):
                    bart_state_dict[k[len('model.'):]] = v

        new_bart_state_dict = self.model.state_dict()
        for x in new_bart_state_dict:
            if x.endswith('embed_positions.weight'
                          ) and self.max_position_embeddings != 1024:
                new_bart_state_dict[x] = torch.nn.Parameter(
                    self.resize_bart_abs_pos_emb(
                        bart_state_dict[x],
                        self.max_position_embeddings + 2,
                    ))
            elif x.endswith('embed_tokens.weight') or x.endswith(
                    'lm_head.weight'):
                new_bart_state_dict[x] = bart_state_dict[x][:len(self.tokenizer
                                                                 ), :]
            else:
                new_bart_state_dict[x] = bart_state_dict[x]
        self.model.load_state_dict(new_bart_state_dict)

    def add_special_tokens(self, list_of_tokens: List[str]):
        """Add special tokens to tokenizer and resize the token embeddings."""
        newly_added_num = self.tokenizer.add_special_tokens(
            {'additional_special_tokens': sorted(set(list_of_tokens))})
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_inputs_for_inference(self,
                                     input_ids: torch.Tensor,
                                     encoder_outputs: torch.Tensor,
                                     past_key_values=None,
                                     past=None,
                                     use_cache: bool = None,
                                     attention_mask: torch.Tensor = None):
        """
        Args:
            input_ids: (batch_size, sequence_lenth)
        Returns:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, embedding_dim)
        """
        # for compatibility with transformers==4.11.x
        if past is not None:
            past_key_values = past
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        output = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'encoder_hidden_states': encoder_outputs.last_hidden_state,
        }
        return output

    def extract_feat(self,
                     input_ids,
                     encoder_hidden_states: Optional[torch.Tensor],
                     attention_mask: Optional[torch.Tensor] = None,
                     past_key_values: Optional[torch.Tensor] = None,
                     use_cache: bool = None,
                     output_attentions: Optional[torch.Tensor] = None,
                     output_hidden_states: Optional[torch.Tensor] = None,
                     return_dict: bool = True,
                     data_samples=None):
        """"""
        if output_attentions is None:
            output_attentions = self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.model.config.output_hidden_states)
        if return_dict is None:
            return_dict = self.model.config.use_return_dict

        outputs = self.model.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.model.lm_head(outputs[0])
        outputs['logits'] = logits
        return outputs

    def loss(self,
             input_ids,
             encoder_hidden_states: Optional[torch.Tensor],
             labels,
             attention_mask: Optional[torch.Tensor] = None,
             past_key_values: Optional[torch.Tensor] = None,
             use_cache: bool = None,
             output_attentions: Optional[torch.Tensor] = None,
             output_hidden_states: Optional[torch.Tensor] = None,
             data_samples=None):
        """A forward function to get cross attentions and utilize `generate`
        function.

        Source:
        https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L1669-L1810

        Args:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, hidden_size)

        Returns:
            loss: (1, )
            logits: (batch_size, sequence_length, hidden_dim)
            hidden_states: (batch_size, sequence_length, hidden_size)
            decoder_attentions: (batch_size, num_heads, sequence_length,
                                 sequence_length)
            cross_attentions: (batch_size, num_heads, sequence_length,
                               sequence_length)
        """
        outputs = self.extract_feat(
            input_ids,
            encoder_hidden_states,
            attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            data_samples=data_samples)
        logits = outputs['logits']

        loss = None
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        return {'loss': loss}

    def forward(self,
                input_ids,
                encoder_hidden_states: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[torch.Tensor] = None,
                use_cache: bool = None,
                output_attentions: Optional[torch.Tensor] = None,
                output_hidden_states: Optional[torch.Tensor] = None,
                return_dict=None,
                data_samples=None):
        """A forward function to get cross attentions and utilize `generate`
        function.

        Source:
        https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L1669-L1810

        Args:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, hidden_size)

        Returns:
            loss: (1, )
            logits: (batch_size, sequence_length, hidden_dim)
            hidden_states: (batch_size, sequence_length, hidden_size)
            decoder_attentions: (batch_size, num_heads,
                                 sequence_length, sequence_length)
            cross_attentions: (batch_size, num_heads,
                               sequence_length, sequence_length)
        """
        outputs = self.extract_feat(
            input_ids,
            encoder_hidden_states,
            attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            data_samples=data_samples)

        return ModelOutput(
            loss=None,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor,
                                max_length: int) -> torch.Tensor:
        """Resize position embeddings Truncate if sequence length of Bart
        backbone is greater than given max_length, else interpolate to
        max_length."""
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode='linear',
                    align_corners=False,
                ).squeeze(0).permute(1, 0))
        return weight
