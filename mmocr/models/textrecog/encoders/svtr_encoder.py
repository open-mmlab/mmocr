# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import DropPath

from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_init
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample


class OverlapPatchEmbed(BaseModule):
    """Image to the progressive overlapping Patch Embedding.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        embed_dims (int): The dimensions of embedding. Defaults to 768.
        num_layers (int, optional): Number of Conv_BN_Layer. Defaults to 2 and
            limit to [2, 3].
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 num_layers: int = 2,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None):

        super().__init__(init_cfg=init_cfg)

        assert num_layers in [2, 3], \
            'The number of layers must belong to [2, 3]'
        self.net = nn.Sequential()
        for num in range(num_layers, 0, -1):
            if (num == num_layers):
                _input = in_channels
            _output = embed_dims // (2**(num - 1))
            self.net.add_module(
                f'ConvModule{str(num_layers - num)}',
                ConvModule(
                    in_channels=_input,
                    out_channels=_output,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='GELU')))
            _input = _output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (Tensor): A Tensor of shape :math:`(N, C, H, W)`.

        Returns:
            Tensor: A tensor of shape math:`(N, HW//16, C)`.
        """
        x = self.net(x).flatten(2).permute(0, 2, 1)
        return x


class ConvMixer(BaseModule):
    """The conv Mixer.

    Args:
        embed_dims (int): Number of character components.
        num_heads (int, optional): Number of heads. Defaults to 8.
        input_shape (Tuple[int, int], optional): The shape of input [H, W].
            Defaults to [8, 25].
        local_k (Tuple[int, int], optional): Window size. Defaults to [3, 3].
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 input_shape: Tuple[int, int] = [8, 25],
                 local_k: Tuple[int, int] = [3, 3],
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None):
        super().__init__(init_cfg)
        self.input_shape = input_shape
        self.embed_dims = embed_dims
        self.local_mixer = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=local_k,
            stride=1,
            padding=(local_k[0] // 2, local_k[1] // 2),
            groups=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, HW, C)`.

        Returns:
            torch.Tensor: Tensor: A tensor of shape math:`(N, HW, C)`.
        """
        h, w = self.input_shape
        x = x.permute(0, 2, 1).reshape([-1, self.embed_dims, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class AttnMixer(BaseModule):
    """One of mixer of {'Global', 'Local'}. Defaults to Global Mixer.

    Args:
        embed_dims (int): Number of character components.
        num_heads (int, optional): Number of heads. Defaults to 8.
        mixer (str, optional): The mixer type, choices are 'Global' and
            'Local'. Defaults to 'Global'.
        input_shape (Tuple[int, int], optional): The shape of input [H, W].
            Defaults to [8, 25].
        local_k (Tuple[int, int], optional): Window size. Defaults to [7, 11].
        qkv_bias (bool, optional): Whether a additive bias is required.
            Defaults to False.
        qk_scale (float, optional): A scaling factor. Defaults to None.
        attn_drop (float, optional): Attn dropout probability. Defaults to 0.0.
        proj_drop (float, optional): Proj dropout layer. Defaults to 0.0.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 mixer: str = 'Global',
                 input_shape: Tuple[int, int] = [8, 25],
                 local_k: Tuple[int, int] = [7, 11],
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None):
        super().__init__(init_cfg)
        assert mixer in {'Global', 'Local'}, \
            "The type of mixer must belong to {'Global', 'Local'}"
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_shape = input_shape
        if input_shape is not None:
            height, width = input_shape
            self.input_size = height * width
            self.embed_dims = embed_dims
        if mixer == 'Local' and input_shape is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones(
                [height * width, height + hk - 1, width + wk - 1],
                dtype=torch.float32)
            for h in range(0, height):
                for w in range(0, width):
                    mask[h * width + w, h:h + hk, w:w + wk] = 0.
            mask = mask[:, hk // 2:height + hk // 2,
                        wk // 2:width + wk // 2].flatten(1)
            mask[mask >= 1] = -np.inf
            self.register_buffer('mask', mask[None, None, :, :])
        self.mixer = mixer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.

        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H, W, C)`.
        """
        if self.input_shape is not None:
            input_size, embed_dims = self.input_size, self.embed_dims
        else:
            _, input_size, embed_dims = x.shape
        qkv = self.qkv(x).reshape((-1, input_size, 3, self.num_heads,
                                   embed_dims // self.num_heads)).permute(
                                       (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q.matmul(k.permute(0, 1, 3, 2))
        if self.mixer == 'Local':
            attn += self.mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn.matmul(v).permute(0, 2, 1, 3).reshape(-1, input_size,
                                                       embed_dims)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(BaseModule):
    """The MLP block.

    Args:
        in_features (int): The input features.
        hidden_features (int, optional): The hidden features.
            Defaults to None.
        out_features (int, optional): The output features.
            Defaults to None.
        drop (float, optional): cfg of dropout function. Defaults to 0.0.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int = None,
                 out_features: int = None,
                 drop: float = 0.,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None):
        super().__init__(init_cfg)
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.

        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H, W, C)`.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixingBlock(BaseModule):
    """The Mixing block.

    Args:
        embed_dims (int): Number of character components.
        num_heads (int): Number of heads
        mixer (str, optional): The mixer type. Defaults to 'Global'.
        window_size (Tuple[int ,int], optional): Local window size.
            Defaults to [7, 11].
        input_shape (Tuple[int, int], optional): The shape of input [H, W].
            Defaults to [8, 25].
        mlp_ratio (float, optional): The ratio of hidden features to input.
            Defaults to 4.0.
        qkv_bias (bool, optional): Whether a additive bias is required.
            Defaults to False.
        qk_scale (float, optional): A scaling factor. Defaults to None.
        drop (float, optional): cfg of Dropout. Defaults to 0..
        attn_drop (float, optional): cfg of Dropout. Defaults to 0.0.
        drop_path (float, optional): The probability of drop path.
            Defaults to 0.0.
        pernorm (bool, optional): Whether to place the MxingBlock before norm.
            Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 mixer: str = 'Global',
                 window_size: Tuple[int, int] = [7, 11],
                 input_shape: Tuple[int, int] = [8, 25],
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path=0.,
                 prenorm: bool = True,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None):
        super().__init__(init_cfg)
        self.norm1 = nn.LayerNorm(embed_dims, eps=1e-6)
        if mixer in {'Global', 'Local'}:
            self.mixer = AttnMixer(
                embed_dims,
                num_heads=num_heads,
                mixer=mixer,
                input_shape=input_shape,
                local_k=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                embed_dims,
                num_heads=num_heads,
                input_shape=input_shape,
                local_k=window_size)
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dims, eps=1e-6)
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = MLP(
            in_features=embed_dims, hidden_features=mlp_hidden_dim, drop=drop)
        self.prenorm = prenorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H*W, C)`.

        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H*W, C)`.
        """
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MerigingBlock(BaseModule):
    """The last block of any stage, except for the last stage.

    Args:
        in_channels (int): The channels of input.
        out_channels (int): The channels of output.
        types (str, optional): Which downsample operation of ['Pool', 'Conv'].
            Defaults to 'Pool'.
        stride (Union[int, Tuple[int, int]], optional): Stride of the Conv.
            Defaults to [2, 1].
        act (bool, optional): activation function. Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 types: str = 'Pool',
                 stride: Union[int, Tuple[int, int]] = [2, 1],
                 act: bool = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None):
        super().__init__(init_cfg)
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2d(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2d(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1)
        self.norm = nn.LayerNorm(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.

        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H/2, W, 2C)`.
        """
        if self.types == 'Pool':
            x = (self.avgpool(x) + self.maxpool(x)) * 0.5
            out = self.proj(x.flatten(2).permute(0, 2, 1))

        else:
            x = self.conv(x)
            out = x.flatten(2).permute(0, 2, 1)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out


@MODELS.register_module()
class SVTREncoder(BaseModule):
    """A PyTorch implementation of `SVTR: Scene Text Recognition with a Single
    Visual Model <https://arxiv.org/abs/2205.00159>`_

    Code is partially modified from https://github.com/PaddlePaddle/PaddleOCR.

    Args:
        img_size (Tuple[int, int], optional): The expected input image shape.
            Defaults to [32, 100].
        in_channels (int, optional): The num of input channels. Defaults to 3.
        embed_dims (Tuple[int, int, int], optional): Number of input channels.
            Defaults to [64, 128, 256].
        depth (Tuple[int, int, int], optional):
            The number of MixingBlock at each stage. Defaults to [3, 6, 3].
        num_heads (Tuple[int, int, int], optional): Number of attention heads.
            Defaults to [2, 4, 8].
        mixer_types (Tuple[str], optional): Mixing type in a MixingBlock.
            Defaults to ['Local']*6+['Global']*6.
        window_size (Tuple[Tuple[int, int]], optional):
            The height and width of the window at eeach stage.
            Defaults to [[7, 11], [7, 11], [7, 11]].
        merging_types (str, optional): The way of downsample in MergingBlock.
            Defaults to 'Conv'.
        mlp_ratio (int, optional): Ratio of hidden features to input in MLP.
            Defaults to 4.
        qkv_bias (bool, optional):
            Whether to add bias for qkv in attention modules. Defaults to True.
        qk_scale (float, optional): A scaling factor. Defaults to None.
        drop_rate (float, optional): Probability of an element to be zeroed.
            Defaults to 0.0.
        last_drop (float, optional): cfg of dropout at last stage.
            Defaults to 0.1.
        attn_drop_rate (float, optional): _description_. Defaults to 0..
        drop_path_rate (float, optional): stochastic depth rate.
            Defaults to 0.1.
        out_channels (int, optional): The num of output channels in backone.
            Defaults to 192.
        max_seq_len (int, optional): Maximum output sequence length :math:`T`.
            Defaults to 25.
        num_layers (int, optional): The num of conv in PatchEmbedding.
            Defaults to 2.
        prenorm (bool, optional): Whether to place the MixingBlock before norm.
            Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 img_size: Tuple[int, int] = [32, 100],
                 in_channels: int = 3,
                 embed_dims: Tuple[int, int, int] = [64, 128, 256],
                 depth: Tuple[int, int, int] = [3, 6, 3],
                 num_heads: Tuple[int, int, int] = [2, 4, 8],
                 mixer_types: Tuple[str] = ['Local'] * 6 + ['Global'] * 6,
                 window_size: Tuple[Tuple[int, int]] = [[7, 11], [7, 11],
                                                        [7, 11]],
                 merging_types: str = 'Conv',
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 last_drop: float = 0.1,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 out_channels: int = 192,
                 max_seq_len: int = 25,
                 num_layers: int = 2,
                 prenorm: bool = True,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None):
        super().__init__(init_cfg)
        self.img_size = img_size
        self.embed_dims = embed_dims
        self.out_channels = out_channels
        self.prenorm = prenorm
        self.patch_embed = OverlapPatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims[0],
            num_layers=num_layers)
        num_patches = (img_size[1] // (2**num_layers)) * (
            img_size[0] // (2**num_layers))
        self.input_shape = [
            img_size[0] // (2**num_layers), img_size[1] // (2**num_layers)
        ]
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros([1, num_patches, embed_dims[0]], dtype=torch.float32),
            requires_grad=True)
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = np.linspace(0, drop_path_rate, sum(depth))

        self.blocks1 = nn.ModuleList([
            MixingBlock(
                embed_dims=embed_dims[0],
                num_heads=num_heads[0],
                mixer=mixer_types[0:depth[0]][i],
                window_size=window_size[0],
                input_shape=self.input_shape,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[0:depth[0]][i],
                prenorm=prenorm) for i in range(depth[0])
        ])
        self.downsample1 = MerigingBlock(
            in_channels=embed_dims[0],
            out_channels=embed_dims[1],
            types=merging_types,
            stride=[2, 1])
        input_shape = [self.input_shape[0] // 2, self.input_shape[1]]
        self.merging_types = merging_types

        self.blocks2 = nn.ModuleList([
            MixingBlock(
                embed_dims=embed_dims[1],
                num_heads=num_heads[1],
                mixer=mixer_types[depth[0]:depth[0] + depth[1]][i],
                window_size=window_size[1],
                input_shape=input_shape,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0]:depth[0] + depth[1]][i],
                prenorm=prenorm) for i in range(depth[1])
        ])
        self.downsample2 = MerigingBlock(
            in_channels=embed_dims[1],
            out_channels=embed_dims[2],
            types=merging_types,
            stride=[2, 1])
        input_shape = [self.input_shape[0] // 4, self.input_shape[1]]

        self.blocks3 = nn.ModuleList([
            MixingBlock(
                embed_dims=embed_dims[2],
                num_heads=num_heads[2],
                mixer=mixer_types[depth[0] + depth[1]:][i],
                window_size=window_size[2],
                input_shape=input_shape,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + depth[1]:][i],
                prenorm=prenorm) for i in range(depth[2])
        ])
        self.layer_norm = nn.LayerNorm(self.embed_dims[-1], eps=1e-6)
        self.avgpool = nn.AdaptiveAvgPool2d([1, max_seq_len])
        self.last_conv = nn.Conv2d(
            in_channels=embed_dims[2],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,
            stride=1,
            padding=0)
        self.hardwish = nn.Hardswish()
        self.dropout = nn.Dropout(p=last_drop)

        trunc_normal_init(self.absolute_pos_embed, mean=0, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function except the last combing operation.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.

        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H/16, W/4, 256)`.
        """
        x = self.patch_embed(x)
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.downsample1(
            x.permute(0, 2, 1).reshape([
                -1, self.embed_dims[0], self.input_shape[0],
                self.input_shape[1]
            ]))

        for blk in self.blocks2:
            x = blk(x)
        x = self.downsample2(
            x.permute(0, 2, 1).reshape([
                -1, self.embed_dims[1], self.input_shape[0] // 2,
                self.input_shape[1]
            ]))

        for blk in self.blocks3:
            x = blk(x)
        if not self.prenorm:
            x = self.layer_norm(x)
        return x

    def forward(self,
                x: torch.Tensor,
                data_samples: List[TextRecogDataSample] = None
                ) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H/16, W/4, 256)`.
            data_samples (list[TextRecogDataSample]): Batch of
                TextRecogDataSample. Defaults to None.

        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, 1, W/4, 192)`.
        """
        x = self.forward_features(x)
        x = self.avgpool(
            x.permute(0, 2, 1).reshape([
                -1, self.embed_dims[2], self.input_shape[0] // 4,
                self.input_shape[1]
            ]))
        x = self.last_conv(x)
        x = self.hardwish(x)
        x = self.dropout(x)
        return x
