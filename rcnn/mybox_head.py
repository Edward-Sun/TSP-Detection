# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
import math
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY
from .transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from .conv_block import MyConvBlock

__all__ = ["MyFastRCNNTransformerHead"]


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


@ROI_BOX_HEAD_REGISTRY.register()
class MyFastRCNNTransformerHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        use_encoder_decoder: bool = False,
        use_position_encoding: bool = False,
        use_linear_attention: bool = False,
        num_conv: int = 0,
        conv_dim: int = 256,
        num_fc: int = 0,
        fc_dim: int = 1024,
        num_self_attention: int = 0,
        self_attention_dim: int = 256,
        visualize: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        self.d_model = d_model
        self._output_size = d_model
        hidden_dim = d_model
        self.num_conv = num_conv
        self.conv = None
        self.self_attn = None
        self.visualize = visualize

        if num_self_attention > 0:
            if num_self_attention > 1:
                raise NotImplementedError
            in_channels = input_shape.channels
            if self_attention_dim != in_channels:
                self.conv = MyConvBlock(in_channels, self_attention_dim, norm=True, activation=False)
            self.self_attn = nn.MultiheadAttention(self_attention_dim, 4, dropout=dropout)
        elif num_conv > 0:
            in_channels = input_shape.channels
            nn_list = ([MyConvBlock(in_channels, conv_dim, norm=True, activation=True)]
                       + [MyConvBlock(conv_dim, conv_dim, norm=True, activation=True) for _ in range(num_conv - 1)])
            self.conv = nn.Sequential(*nn_list)

        self.input_proj = None
        if num_fc >= 1:
            if num_self_attention > 0 and self_attention_dim != input_shape.channels:
                total_channels = self_attention_dim * input_shape.height * input_shape.width
            elif num_conv > 0:
                total_channels = conv_dim * input_shape.height * input_shape.width
            else:
                total_channels = input_shape.channels * input_shape.height * input_shape.width
            if num_fc == 1:
                self.input_proj = nn.Linear(total_channels, hidden_dim)
            else:
                nn_list = [nn.Linear(total_channels, fc_dim)]
                for i in range(num_fc - 2):
                    nn_list.extend([nn.ReLU(inplace=True), nn.Linear(fc_dim, fc_dim)])
                nn_list.extend([nn.ReLU(inplace=True), nn.Linear(fc_dim, hidden_dim)])
                self.input_proj = nn.Sequential(*nn_list)

        self.post_input_proj_norm = nn.LayerNorm(d_model)
        self.use_encoder_decoder = use_encoder_decoder
        self.use_position_encoding = use_position_encoding
        if use_encoder_decoder:
            self.enc_proj = nn.Linear(input_shape.channels, hidden_dim)
            self.post_enc_proj_norm = nn.LayerNorm(d_model)
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before, use_linear_attention)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before, use_linear_attention)
            decoder_norm = nn.LayerNorm(d_model)
            self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                                          return_intermediate=False)
            self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before, use_linear_attention)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        if use_position_encoding:
            self.dec_pos_embed_proj = nn.Linear(hidden_dim, hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            "visualize": cfg.MODEL.VISUALIZE,
            "input_shape": input_shape,
            "d_model": cfg.MODEL.MY_ROI_BOX_HEAD.D_MODEL,
            "nhead": cfg.MODEL.MY_ROI_BOX_HEAD.NHEAD,
            "num_encoder_layers": cfg.MODEL.MY_ROI_BOX_HEAD.NUM_ENCODER_LAYERS,
            "num_decoder_layers": cfg.MODEL.MY_ROI_BOX_HEAD.NUM_DECODER_LAYERS,
            "dim_feedforward": cfg.MODEL.MY_ROI_BOX_HEAD.DIM_FEEDFORWARD,
            "dropout": cfg.MODEL.MY_ROI_BOX_HEAD.DROPOUT,
            "activation": cfg.MODEL.MY_ROI_BOX_HEAD.ACTIVATION,
            "normalize_before": cfg.MODEL.MY_ROI_BOX_HEAD.NORMALIZE_BEFORE,
            "use_encoder_decoder": cfg.MODEL.MY_ROI_BOX_HEAD.USE_ENCODER_DECODER,
            "use_position_encoding": cfg.MODEL.MY_ROI_BOX_HEAD.USE_POSITION_ENCODING,
            "use_linear_attention": cfg.MODEL.MY_ROI_BOX_HEAD.USE_LINEAR_ATTENTION,
            "num_conv": cfg.MODEL.MY_ROI_BOX_HEAD.NUM_CONV,
            "conv_dim": cfg.MODEL.MY_ROI_BOX_HEAD.CONV_DIM,
            "num_fc": cfg.MODEL.MY_ROI_BOX_HEAD.NUM_FC,
            "fc_dim": cfg.MODEL.MY_ROI_BOX_HEAD.FC_DIM,
            "num_self_attention": cfg.MODEL.MY_ROI_BOX_HEAD.NUM_SELF_ATTENTION,
            "self_attention_dim": cfg.MODEL.MY_ROI_BOX_HEAD.SELF_ATTENTION_DIM,
        }
        return ret

    def forward(self, enc_feature, enc_mask, x, dec_mask, proposals, prev_box_features=None):
        batch_size, seq_length, n_channels, nh, nw = x.shape
        if self.self_attn is not None:
            x_conv = None
            if self.conv is not None:
                x = self.conv(x.view(batch_size * seq_length, n_channels, nh, nw))
                x_conv = x
            x = x.view(batch_size * seq_length, -1, nh * nw).permute(2, 0, 1)
            x = self.self_attn(x, x, x, need_weights=False)[0]
            x = x.view(nh * nw, batch_size, seq_length, -1).permute(2, 1, 0, 3)
            x_conv = x_conv.view(batch_size, seq_length, -1, nh * nw).permute(1, 0, 3, 2)
            x = x + x_conv
        elif self.conv is not None:
            x = self.conv(x.view(batch_size * seq_length, n_channels, nh, nw))
            x = x.view(batch_size, seq_length, -1)
            x = x.transpose(0, 1).contiguous()
        else:
            x = x.transpose(0, 1).contiguous()
        x = x.flatten(2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        hidden_size = x.shape[-1]
        bbox_pos_embed = None
        if self.use_position_encoding:
            num_pos_feats = self.d_model // 8
            temperature = 10000.0
            dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
            dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

            bbox_pos_embed = (proposals[:, :, :, None] * 2 * math.pi) / dim_t
            bbox_pos_embed = torch.stack((bbox_pos_embed[:, :, :, 0::2].sin(),
                                          bbox_pos_embed[:, :, :, 1::2].cos()), dim=4).flatten(2)
            bbox_pos_embed = bbox_pos_embed.transpose(0, 1)
            dec_pos_embed = self.dec_pos_embed_proj(bbox_pos_embed)
            x = x + dec_pos_embed

        x = self.post_input_proj_norm(x)

        if prev_box_features is not None:
            prev_box_features = prev_box_features.view(batch_size, seq_length, hidden_size).transpose(0, 1)
            x = (x + prev_box_features) / (2**0.5)

        attention_maps = None
        if self.use_encoder_decoder:
            enc_pos_embed = self.position_embedding(enc_feature, enc_mask)
            enc_pos_embed = enc_pos_embed.flatten(2).permute(2, 0, 1)
            enc_feature = enc_feature.flatten(2).permute(2, 0, 1)
            enc_feature = self.enc_proj(enc_feature)
            enc_feature = self.post_enc_proj_norm(enc_feature)
            enc_mask = enc_mask.flatten(1)

            memory = self.transformer_encoder(enc_feature, src_key_padding_mask=enc_mask, pos=enc_pos_embed)
            x = self.transformer_decoder(x, memory, memory_key_padding_mask=enc_mask, tgt_key_padding_mask=dec_mask,
                                         pos=enc_pos_embed, query_pos=bbox_pos_embed)
        else:
            if self.visualize:
                x, attention_maps = self.transformer_encoder(x, src_key_padding_mask=dec_mask, pos=bbox_pos_embed, return_attention_maps=True)
            else:
                x = self.transformer_encoder(x, src_key_padding_mask=dec_mask, pos=bbox_pos_embed)

        x = x.transpose(0, 1).contiguous().view(batch_size * seq_length, hidden_size)
        if self.visualize:
            return x, attention_maps
        else:
            return x

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            raise NotImplementedError
