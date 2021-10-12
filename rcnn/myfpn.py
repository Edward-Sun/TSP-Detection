# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone

from .conv_block import MyConvBlock

__all__ = ["build_resnet_mybifpn_backbone", "build_resnet_myfpn_backbone_v2", "MYFPN",
           "build_resnet_myfpn_backbone", "build_resnet_mybifpn_backbone_v2", "MYBIFPN",
           "build_resnet_myfpn_backbone_p4"]


class MYBIFPN(Backbone):
    def __init__(
        self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum", num_repeats=2,
    ):
        super(MYBIFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)

        self.num_repeats = num_repeats

        up_convs = [[] for _ in range(num_repeats)]
        up_convs_extra = [[] for _ in range(num_repeats)]
        down_convs = [[] for _ in range(num_repeats)]
        affines_1 = [[] for _ in range(num_repeats)]
        affines_2 = [[] for _ in range(num_repeats)]

        extra_num_levels = 0
        if top_block is not None:
            extra_num_levels = top_block.num_levels
        for repeat in range(num_repeats):
            for idx in range(len(in_channels) + extra_num_levels):
                # res2 ~ res5
                if idx > 0:
                    down_conv = MyConvBlock(out_channels, out_channels)
                    down_convs[repeat].append(down_conv)
                    self.add_module("bifpn_down_{}_{}".format(repeat, idx), down_conv)
                if idx < len(in_channels) + extra_num_levels - 1:
                    up_conv = MyConvBlock(out_channels, out_channels)
                    up_convs[repeat].append(up_conv)
                    self.add_module("bifpn_up_{}_{}".format(repeat, idx), up_conv)
                    up_conv_extra = MyConvBlock(out_channels, out_channels, stride=2)
                    up_convs_extra[repeat].append(up_conv_extra)
                    self.add_module("bifpn_up_extra_{}_{}".format(repeat, idx), up_conv_extra)

                if idx < len(in_channels) + extra_num_levels - 1:
                    affine_weight = nn.Parameter(torch.ones(2, out_channels, 1, 1), requires_grad=True)
                    affines_1[repeat].append(affine_weight)
                    self.register_parameter("bifpn_affine_1_{}_{}".format(repeat, idx), affine_weight)

                if idx > 0:
                    if idx < len(in_channels) + extra_num_levels - 1:
                        affine_weight = nn.Parameter(torch.ones(3, out_channels, 1, 1), requires_grad=True)
                    else:
                        affine_weight = nn.Parameter(torch.ones(2, out_channels, 1, 1), requires_grad=True)
                    affines_2[repeat].append(affine_weight)
                    self.register_parameter("bifpn_affine_2_{}_{}".format(repeat, idx), affine_weight)

        self.up_convs = up_convs
        self.up_convs_extra = up_convs_extra
        self.down_convs = down_convs
        self.affines_1 = affines_1
        self.affines_2 = affines_2

        first_time_convs = []
        for idx, in_channel in enumerate(in_channels):
            first_time_conv = MyConvBlock(in_channel, out_channels, kernel_size=1)
            first_time_convs.append(first_time_conv)
            self.add_module("bifpn_ft_{}".format(idx), first_time_conv)
        self.first_time_convs = first_time_convs

        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        # top block output feature maps.
        stage = int(math.log2(in_strides[-1]))
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        # print("bottom_up")
        # print(self.bottom_up.stem.conv1.weight)
        # print(self.bottom_up.stem.conv1.norm.weight)
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        first_time_features = []
        for first_time_conv, in_feature in zip(self.first_time_convs, self.in_features):
            first_time_feature = first_time_conv(bottom_up_features[in_feature])
            first_time_features.append(first_time_feature)

        if self.top_block is not None:
            top_block_in_feature = first_time_features[-1]
            first_time_feature_list = self.top_block(top_block_in_feature)
            first_time_features.extend(first_time_feature_list)

        num_features = len(first_time_features)
        old_features = first_time_features
        for repeat in range(self.num_repeats):
            inter_features = []
            for i in reversed(range(num_features - 1)):
                if i == num_features - 2:
                    input_1 = old_features[i + 1]
                else:
                    input_1 = inter_features[-1]
                input_2 = old_features[i]
                input_1 = F.interpolate(input_1, size=input_2.shape[-2:], mode="nearest")
                affine = self.affines_1[repeat][i].softmax(0)
                input_sum = affine[0] * input_1 + affine[1] * input_2
                feature = self.down_convs[repeat][i](input_sum)
                inter_features.append(feature)
            inter_features = inter_features[::-1]

            new_features = [inter_features[0]]
            for i in range(1, num_features):
                input_1 = new_features[-1]
                input_1 = self.up_convs_extra[repeat][i - 1](input_1)
                input_2 = old_features[i]
                affine = self.affines_2[repeat][i - 1].softmax(0)
                if i < num_features - 1:
                    input_3 = inter_features[i]
                    input_sum = affine[0] * input_1 + affine[1] * input_2 + affine[2] * input_3
                else:
                    input_sum = affine[0] * input_1 + affine[1] * input_2
                feature = self.up_convs[repeat][i - 1](input_sum)
                new_features.append(feature)
            old_features = new_features
        results = old_features
        ret = dict(zip(self._out_features, results))
        ret.update(bottom_up_features)
        return ret

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class MYFPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum", use_p4=False,
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(MYFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(in_strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            if (not use_p4) or (use_p4 and stage == 4):
                self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        if use_p4:
            self.output_convs[0] = None
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # print("bottom_up")
        # print(self.bottom_up.stem.conv1.weight)
        # print(self.bottom_up.stem.conv1.norm.weight)
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        if self.output_convs[0] is not None:
            results.append(self.output_convs[0](prev_features))
        else:
            results.append(None)
        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        ret = dict(zip(self._out_features, results))
        ret.update(bottom_up_features)
        return ret

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


@BACKBONE_REGISTRY.register()
def build_resnet_myfpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = MYFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


class LastLevelMaxPoolV2(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self, channels):
        super().__init__()
        self.num_levels = 2
        self.in_feature = "p5"
        self.p6 = MyConvBlock(channels, channels, stride=2)
        self.p7 = MyConvBlock(channels, channels, stride=2)

    def forward(self, x):
        x1 = self.p6(x)
        x2 = self.p7(F.relu(x1))
        return [x1, x2]


@BACKBONE_REGISTRY.register()
def build_resnet_myfpn_backbone_v2(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = MYFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPoolV2(out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


class MyLastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 2
        self.in_feature = "res5"

    def forward(self, x):
        x1 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x2 = F.max_pool2d(x1, kernel_size=3, stride=2, padding=1)
        return [x1, x2]


@BACKBONE_REGISTRY.register()
def build_resnet_mybifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = MYBIFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=MyLastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        num_repeats=cfg.MODEL.FPN.NUM_REPEATS,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_resnet_mybifpn_backbone_v2(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = MYBIFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPoolV2(out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        num_repeats=cfg.MODEL.FPN.NUM_REPEATS,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_resnet_myfpn_backbone_p4(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = MYFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=None,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        use_p4=True,
    )
    return backbone
