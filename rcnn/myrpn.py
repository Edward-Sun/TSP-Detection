# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.proposal_generator.rpn import RPN_HEAD_REGISTRY
from detectron2.utils import env
from detectron2.layers.batch_norm import NaiveSyncBatchNorm

from .conv_block import MyConvBlock

__all__ = ["MyStandardRPNHead"]


@RPN_HEAD_REGISTRY.register()
class MyStandardRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    @configurable
    def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4,
                 num_conv: int = 1, pyramid_levels: int = 1):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
        """
        super().__init__()
        # 3x3 conv for the hidden representation
        self.obj_conv = nn.ModuleList(
            [MyConvBlock(in_channels, in_channels, norm=False, activation=False) for _ in range(num_conv)])
        self.anchor_conv = nn.ModuleList(
            [MyConvBlock(in_channels, in_channels, norm=False, activation=False) for _ in range(num_conv)])

        SyncBN = NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm

        self.obj_bn_list = nn.ModuleList(
            [nn.ModuleList([SyncBN(in_channels) for i in range(num_conv)]) for j in range(pyramid_levels)])
        self.anchor_bn_list = nn.ModuleList(
            [nn.ModuleList([SyncBN(in_channels) for i in range(num_conv)]) for j in range(pyramid_levels)])

        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        for l in [self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {"in_channels": in_channels,
                "num_anchors": num_anchors[0],
                "box_dim": box_dim,
                "num_conv": cfg.MODEL.RPN.NUM_CONV,
                "pyramid_levels": len(input_shape)}

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps
        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []

        for x, obj_bn, anchor_bn in zip(features, self.obj_bn_list, self.anchor_bn_list):
            t_obj = x
            for bn, conv in zip(obj_bn, self.obj_conv):
                t_obj = conv(t_obj)
                t_obj = bn(t_obj)
                t_obj = F.relu(t_obj, inplace=True)

            t_anchor = x
            for bn, conv in zip(anchor_bn, self.anchor_conv):
                t_anchor = conv(t_anchor)
                t_anchor = bn(t_anchor)
                t_anchor = F.relu(t_anchor, inplace=True)

            pred_objectness_logits.append(self.objectness_logits(t_obj))
            pred_anchor_deltas.append(self.anchor_deltas(t_anchor))
        return pred_objectness_logits, pred_anchor_deltas
