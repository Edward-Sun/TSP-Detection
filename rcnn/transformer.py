# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torchvision
from .my_attention import my_multi_head_attention_forward


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, faster=False, second_decoder=False):
        super().__init__()
        self.second_decoder = second_decoder
        if not self.second_decoder:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before, faster)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.faster = faster

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src_shape = src.shape

        enc_self_mask = mask
        boxes = None
        if self.faster:
            enc_self_mask = mask.new_full((bs, 16, 16), False)
            boxes = []
            for i in range(bs):
                roi = torch.nonzero(torch.logical_not(mask[i]))
                roi_x1 = torch.min(roi[:, 1])
                roi_y1 = torch.min(roi[:, 0])
                roi_x2 = torch.max(roi[:, 1])
                roi_y2 = torch.max(roi[:, 0])
                boxes.append([i, roi_x1, roi_y1, roi_x2, roi_y2])
            boxes = torch.FloatTensor(boxes).to(mask.device)

        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if len(query_embed.shape) == 2:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        else:
            query_embed = query_embed.transpose(0, 1)
        enc_self_mask = enc_self_mask.flatten(1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)

        if self.second_decoder:
            memory = src
        else:
            memory = self.encoder(src, src_key_padding_mask=enc_self_mask,
                                  pos=pos_embed, src_shape=src_shape, boxes=boxes)

        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                src_shape: Optional[List] = None,
                boxes: Optional[Tensor] = None,
                return_attention_maps: bool = False):
        output = src

        attention_maps = []
        for layer in self.layers:
            if return_attention_maps:
                output, attention_map = layer(output, src_mask=mask,
                                              src_key_padding_mask=src_key_padding_mask,
                                              pos=pos, src_shape=src_shape, boxes=boxes,
                                              return_attention_maps=return_attention_maps)
                attention_maps.append(attention_map)
            else:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask,
                               pos=pos, src_shape=src_shape, boxes=boxes)

        if self.norm is not None:
            output = self.norm(output)

        if return_attention_maps:
            attention_maps = torch.cat(attention_maps, dim=1)
            return output, attention_maps
        else:
            return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, faster=False, use_linear_attention=False):
        super().__init__()
        self.faster = faster
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        if self.faster:
            self.linear1 = nn.Linear(d_model, dim_feedforward // 4)
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.linear2 = nn.Linear(dim_feedforward // 4, d_model)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     src_shape: Optional[List] = None,
                     boxes: Optional[Tensor] = None,
                     return_attention_maps: bool = False):
        attention_weights = None
        if self.faster:
            bs, c, h, w = src_shape
            src_value = src

            src_value = src_value.permute(1, 2, 0).view(bs, c, h, w)
            src_value = torchvision.ops.roi_align(src_value, boxes, (16, 16), aligned=True)
            src_value = src_value.flatten(2).permute(2, 0, 1)

            pos2 = pos.permute(1, 2, 0).view(bs, c, h, w)
            pos2 = torchvision.ops.roi_align(pos2, boxes, (16, 16), aligned=True)
            pos2 = pos2.flatten(2).permute(2, 0, 1)

            q = self.with_pos_embed(src, pos)
            k = self.with_pos_embed(src_value, pos2)
            src2 = self.self_attn(q, k, value=src_value, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask,
                                  need_weights=False)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        else:
            q = k = self.with_pos_embed(src, pos)
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask,
                                  need_weights=False)[0]
            if return_attention_maps:
                attention_weights = my_multi_head_attention_forward(
                    q, k, src, self.self_attn.embed_dim, self.self_attn.num_heads,
                    self.self_attn.in_proj_weight, self.self_attn.in_proj_bias,
                    self.self_attn.bias_k, self.self_attn.bias_v, self.self_attn.add_zero_attn,
                    self.self_attn.dropout, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias,
                    training=self.self_attn.training,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=True,
                    attn_mask=src_mask)[1]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        if return_attention_maps:
            return src, attention_weights
        else:
            return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    return_attention_maps: bool = False):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,
                              need_weights=False)[0]
        attention_weights = None
        if return_attention_maps:
            attention_weights = my_multi_head_attention_forward(
                q, k, src, self.self_attn.embed_dim, self.self_attn.num_heads,
                self.self_attn.in_proj_weight, self.self_attn.in_proj_bias,
                self.self_attn.bias_k, self.self_attn.bias_v,
                self.self_attn.add_zero_attn,
                self.self_attn.dropout, self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                training=self.self_attn.training,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                attn_mask=src_mask)[1]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if return_attention_maps:
            return src, attention_weights
        else:
            return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                src_shape: Optional[List] = None,
                boxes: Optional[List] = None,
                return_attention_maps: bool = False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attention_maps=return_attention_maps)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, src_shape, boxes, return_attention_maps=return_attention_maps)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_linear_attention=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.dropout3 = nn.Dropout(dropout, inplace=True)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   need_weights=False)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   need_weights=False)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args, second_decoder=False):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        faster=args.faster,
        second_decoder=second_decoder
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
