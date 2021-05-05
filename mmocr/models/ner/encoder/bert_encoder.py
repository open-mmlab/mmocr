import torch
import torch.nn as nn
from mmcv.cnn import uniform_init, xavier_init

from mmocr.models.builder import ENCODERS
from mmocr.models.ner.utils.bert import BertModel


@ENCODERS.register_module()
class BertEncoder(nn.Module):
    """Bert encoder
    Args:
        num_hidden_layers (int): The number of hidden layers.
        initializer_range (float):
        vocab_size (int): Number of words supported.
        hidden_size (int): Hidden size.
        max_position_embeddings (int): Max positionsembedding size.
        type_vocab_size (int): The size of type_vocab.
        layer_norm_eps (float): eps.
        hidden_dropout_prob (float): The dropout probability of hidden layer.
        output_attentions (bool):  Whether use the attentions in output
        output_hidden_states (bool): Whether use the hidden_states in output
        num_attention_heads (int): The number of attention heads.
        attention_probs_dropout_prob (float): The dropout probability
            of attention.
        intermediate_size (int): The size of intermediate layer.
        hidden_act (str):  hidden layer activation
    """

    def __init__(self,
                 num_hidden_layers=12,
                 initializer_range=0.02,
                 vocab_size=21128,
                 hidden_size=768,
                 max_position_embeddings=128,
                 type_vocab_size=2,
                 layer_norm_eps=1e-12,
                 hidden_dropout_prob=0.1,
                 output_attentions=False,
                 output_hidden_states=False,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.1,
                 intermediate_size=3072,
                 hidden_act='gelu_new'):
        super().__init__()
        self.bert = BertModel(
            num_hidden_layers=num_hidden_layers,
            initializer_range=initializer_range,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act)
        self.init_weights()

    def forward(self, img_metas):
        device = next(self.bert.parameters()).device
        input_ids = []
        labels = []
        attention_masks = []
        token_type_ids = []
        for i, _ in enumerate(img_metas):
            input_id = torch.tensor(img_metas[i]['input_ids']).to(device)
            label = torch.tensor(img_metas[i]['labels']).to(device)
            attention_mask = torch.tensor(
                img_metas[i]['attention_mask']).to(device)
            token_type_id = torch.tensor(
                img_metas[i]['token_type_ids']).to(device)
            input_ids.append(input_id)
            labels.append(label)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)

        input_ids = torch.stack(input_ids, 0)
        labels = torch.stack(labels, 0)
        attention_masks = torch.stack(attention_masks, 0)
        token_type_ids = torch.stack(token_type_ids, 0)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids)
        return outputs

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                uniform_init(m)
