import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from detectron2.modeling.backbone import build_backbone
from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.postprocessing import detector_postprocess as d2_postprocesss
from detectron2.layers import Conv2d
from .my_fcos_outputs import FCOSOutputs
from .transformer import TransformerEncoderLayer, TransformerEncoder

__all__ = ["MyFCOS"]

INF = 100000000


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


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    In addition to the post processing of detectron2, we add scalign for
    bezier control points.
    """
    scale_x, scale_y = (output_width / (0.0 + results.image_size[1]), output_height / (0.0 + results.image_size[0]))
    results = d2_postprocesss(results, output_height, output_width, mask_threshold)

    # scale bezier points
    if results.has("beziers"):
        beziers = results.beziers
        # scale and clip in place
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)

    return results


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


class MLP(nn.Module):
  """ Very simple multi-layer perceptron (also called FFN)"""

  def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
    super().__init__()
    self.num_layers = num_layers
    h = [hidden_dim] * (num_layers - 1)
    self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    nn.init.normal_(self.layers[-1].weight, std=0.001)
    for l in [self.layers[-1]]:
        nn.init.constant_(l.bias, 0)

  def forward(self, x):
    for i, layer in enumerate(self.layers):
      x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
    return x


@META_ARCH_REGISTRY.register()
class MyFCOS(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1904.01355).
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()

        self.visualize = cfg.MODEL.VISUALIZE
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

        backbone_shapes = [backbone_shape[f] for f in self.in_features]
        self.fcos_head = FCOSHead(cfg, backbone_shapes)
        self.in_channels_to_top_module = self.fcos_head.in_channels_to_top_module
        self.num_classes = self.fcos_head.num_classes
        self.num_levels = self.fcos_head.num_levels

        self.fcos_outputs = FCOSOutputs(cfg)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        d_model = cfg.MODEL.MY_ROI_BOX_HEAD.D_MODEL
        nhead = cfg.MODEL.MY_ROI_BOX_HEAD.NHEAD
        num_encoder_layers = cfg.MODEL.MY_ROI_BOX_HEAD.NUM_ENCODER_LAYERS
        dim_feedforward = cfg.MODEL.MY_ROI_BOX_HEAD.DIM_FEEDFORWARD
        dropout = cfg.MODEL.MY_ROI_BOX_HEAD.DROPOUT
        activation = cfg.MODEL.MY_ROI_BOX_HEAD.ACTIVATION
        normalize_before = cfg.MODEL.MY_ROI_BOX_HEAD.NORMALIZE_BEFORE
        use_linear_attention = cfg.MODEL.MY_ROI_BOX_HEAD.USE_LINEAR_ATTENTION

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, use_linear_attention)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.position_embedding = PositionEmbeddingSine(d_model // 2, normalize=True)
        self.pyramid_position_embedding = nn.Parameter(torch.ones(len(self.in_features), d_model), requires_grad=True)

        self.enc_proj = nn.Linear(backbone_shapes[0].channels * 2, d_model)
        self.post_enc_proj_norm = nn.LayerNorm(d_model)

        self.num_proposal = cfg.MODEL.FCOS.NUM_PROPOSAL
        self.random_proposal_drop = cfg.MODEL.FCOS.RANDOM_PROPOSAL_DROP
        self.random_proposal_drop_upper_bound = cfg.MODEL.FCOS.RANDOM_PROPOSAL_DROP_UPPER_BOUND
        self.random_proposal_drop_lower_bound = cfg.MODEL.FCOS.RANDOM_PROPOSAL_DROP_LOWER_BOUND
        self.random_sample_size = cfg.MODEL.FCOS.RANDOM_SAMPLE_SIZE
        self.random_sample_size_upper_bound = cfg.MODEL.FCOS.RANDOM_SAMPLE_SIZE_UPPER_BOUND
        self.random_sample_size_lower_bound = cfg.MODEL.FCOS.RANDOM_SAMPLE_SIZE_LOWER_BOUND

        self.cls_logits = nn.Linear(d_model, self.num_classes)
        # self.bbox_pred = MLP(d_model, d_model, 4, 3)
        self.bbox_pred = nn.Linear(d_model, 4)
        self.ctrness = nn.Linear(d_model, 1)

        if cfg.MODEL.FCOS.USE_SCALE:
            self.scales = nn.Parameter(torch.FloatTensor([1.0] * self.num_levels))
        else:
            self.scales = None

        for module in [self.cls_logits, self.ctrness]:
            for l in module.modules():
                torch.nn.init.normal_(l.weight, std=0.01)
                torch.nn.init.constant_(l.bias, 0)

        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        del batched_inputs
        del results
        return

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            elif "targets" in batched_inputs[0]:
                gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            extras, losses = self._forward(images, features, gt_instances)
            return losses
        else:
            results, losses = self._forward(images, features)
            processed_results = self._postprocess(results, batched_inputs, images.image_sizes)
            return processed_results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def _postprocess(self, instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            if self.visualize:
                r = results_per_image
            else:
                r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _forward(self, images, raw_features, gt_instances=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        flag = images.image_sizes[0][0] - images.image_sizes[0][1]
        for h, w in images.image_sizes:
            assert (flag * (h - w)) >= 0, str(images.image_sizes)

        features = [raw_features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        local_logits_pred, top_feats, cls_towers, bbox_towers = self.fcos_head(features)

        # Start of Transformer part
        with torch.no_grad():
            b = len(images.image_sizes)
            h = max([x[0] for x in images.image_sizes])
            w = max([x[1] for x in images.image_sizes])
            enc_mask = torch.ones((b, h, w), dtype=torch.bool, device=features[0].device)
            for c, image_size in enumerate(images.image_sizes):
                enc_mask[c, :image_size[0], :image_size[1]] = False
            names = ["res1", "res2"] + self.in_features
            enc_mask_list = {}
            for name in names:
                if name == "res1":
                    target_shape = ((h+1)//2, (w+1)//2)
                else:
                    x = raw_features[name]
                    target_shape = x.shape[-2:]
                m = enc_mask
                enc_mask = F.interpolate(m[None].float(), size=target_shape).to(torch.bool)[0]
                enc_mask_list[name] = enc_mask
            enc_mask_list = [enc_mask_list[f] for f in self.in_features]

            objness_pred = [logits.sigmoid().max(dim=1, keepdim=True)[0] for logits in local_logits_pred]
            if self.training:
                training_targets = self.fcos_outputs.get_ground_truth(locations, gt_instances)
                for fpn_level in range(len(objness_pred)):
                    has_gt = training_targets["labels"][fpn_level] != self.num_classes
                    objness_pred[fpn_level] = torch.where(has_gt.view(objness_pred[fpn_level].shape),
                                                          0.5 * (objness_pred[fpn_level]
                                                                 + torch.ones_like(objness_pred[fpn_level])),
                                                          objness_pred[fpn_level])

        scores = []
        transformer_features = []
        position_encodings = []
        fpn_levels = []
        for fpn_level in range(len(objness_pred)):
            objness = objness_pred[fpn_level]
            enc_mask = enc_mask_list[fpn_level]
            cls_tower = cls_towers[fpn_level]
            bbox_tower = bbox_towers[fpn_level]
            b, _, h, w = objness.shape
            joint_feature = torch.cat([cls_tower, bbox_tower], dim=1)
            enc_pos_embed = self.position_embedding(joint_feature, enc_mask)
            # B, C, H, W
            enc_pos_embed = enc_pos_embed.flatten(2).permute(0, 2, 1)
            enc_pos_embed = enc_pos_embed + self.pyramid_position_embedding[fpn_level]
            # B, H * W, C
            joint_feature = joint_feature.flatten(2).permute(0, 2, 1)
            # B, H * W, C
            objness = objness.view(b, h * w)
            # B, H * W

            scores.append(objness)
            transformer_features.append(joint_feature)
            position_encodings.append(enc_pos_embed)
            fpn_levels.append(torch.full(objness.shape, fpn_level, dtype=torch.int64, device=objness.device))

        scores = torch.cat(scores, dim=1)
        transformer_features = torch.cat(transformer_features, dim=1)
        position_encodings = torch.cat(position_encodings, dim=1)
        fpn_levels = torch.cat(fpn_levels, dim=1)
        b, num_total_proposal, c = transformer_features.shape

        num_proposal = self.num_proposal
        if self.random_sample_size and self.training:
            diff = self.random_sample_size_upper_bound - self.random_sample_size_lower_bound
            sample_factor = self.random_sample_size_upper_bound - np.random.rand(1)[0] * diff
            num_proposal = int(num_proposal * sample_factor)

        _, poi_idx = torch.topk(scores, num_proposal, dim=1)

        if self.random_proposal_drop and self.training:
            diff = self.random_proposal_drop_upper_bound - self.random_proposal_drop_lower_bound
            sample_factor = self.random_proposal_drop_upper_bound - np.random.rand(1)[0] * diff
            original_num_proposal = num_proposal
            num_proposal = int(original_num_proposal * sample_factor)
            subsample_idxs = np.random.choice(original_num_proposal, num_proposal, replace=False)
            subsample_idxs = torch.from_numpy(subsample_idxs).to(poi_idx.device)
            poi_idx = torch.gather(poi_idx, 1, subsample_idxs.view(1, num_proposal).repeat(b, 1))

        poi_idx_flatten = poi_idx + torch.arange(b, dtype=torch.int64, device=poi_idx.device).view(-1, 1) * num_total_proposal
        poi_idx_flatten = poi_idx_flatten.view(-1)

        transformer_features = transformer_features.view(b * num_total_proposal, -1)[poi_idx_flatten]
        transformer_features = transformer_features.view(b, num_proposal, -1)
        position_encodings = position_encodings.view(b * num_total_proposal, -1)[poi_idx_flatten]
        position_encodings = position_encodings.view(b, num_proposal, -1)
        fpn_levels = torch.gather(fpn_levels, 1, poi_idx)

        transformer_features = self.post_enc_proj_norm(self.enc_proj(transformer_features))

        attention_maps = None
        if self.visualize:
            memory, attention_maps = self.transformer_encoder(transformer_features.permute(1, 0, 2),
                                                              pos=position_encodings.permute(1, 0, 2),
                                                              return_attention_maps=True)
        else:
            memory = self.transformer_encoder(transformer_features.permute(1, 0, 2),
                                              pos=position_encodings.permute(1, 0, 2))
        memory = memory.permute(1, 0, 2)
        # bs, #proposal, c

        logits_pred = self.cls_logits(memory)
        ctrness_pred = self.ctrness(memory)
        reg_pred = F.relu(self.bbox_pred(memory))
        if self.scales is not None:
            reg_pred = reg_pred * self.scales[fpn_levels.view(-1)].view(b, num_proposal, -1)
        # End of Transformer part

        if self.training:
            logits_pred = logits_pred.view(b * num_proposal, -1)
            ctrness_pred = ctrness_pred.view(b * num_proposal, -1)
            reg_pred = reg_pred.view(b * num_proposal, -1)
            extras, losses = self.fcos_outputs.losses(
                logits_pred, reg_pred, ctrness_pred,
                locations, gt_instances, top_feats, local_logits_pred, poi_idx_flatten, b
            )
            return extras, losses
        else:
            locations = torch.cat(locations, dim=0).view(1, -1, 2).repeat(b, 1, 1)
            locations = locations.view(b * num_total_proposal, -1)[poi_idx_flatten]
            locations = locations.view(b, num_proposal, 2)

            logits_pred_list = []
            reg_pred_list = []
            ctrness_pred_list = []
            locations_list = []
            attention_maps_list = []
            for fpn_level in range(self.num_levels):
                fpn_idx = (fpn_levels == fpn_level).view(b, num_proposal, 1)
                logits_pred_per_level = torch.where(fpn_idx, logits_pred, torch.ones_like(logits_pred) * -INF)
                logits_pred_list.append(logits_pred_per_level)
                reg_pred_list.append(reg_pred)
                ctrness_pred_list.append(ctrness_pred)
                locations_list.append(locations)
                attention_maps_list.append(attention_maps)

            results = self.fcos_outputs.predict_proposals(
                logits_pred_list, reg_pred_list, ctrness_pred_list,
                locations_list, images.image_sizes, top_feats,
                attention_maps=attention_maps_list,
                visualize=self.visualize,
                fpn_levels=fpn_levels,
            )

            return results, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self._compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    @staticmethod
    def _compute_locations(h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.use_deformable = any(cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE)
        head_configs = {"cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS, self.use_deformable),
                        "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS, self.use_deformable),
                        "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS, False)}
        norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM
        self.num_levels = len(input_shape)

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.in_channels_to_top_module = in_channels

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d
                tower.append(conv_func(
                    in_channels, in_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "BN":
                    tower.append(ModuleListDial([
                        nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)
                    ]))
                elif norm == "SyncBN":
                    tower.append(ModuleListDial([
                        NaiveSyncBatchNorm(in_channels) for _ in range(self.num_levels)
                    ]))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )

        for modules in [self.cls_tower, self.bbox_tower, self.share_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        top_feats = []
        cls_towers = []
        bbox_towers = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            cls_towers.append(cls_tower)
            bbox_towers.append(bbox_tower)
            logits.append(self.cls_logits(cls_tower))
        return logits, top_feats, cls_towers, bbox_towers


class DFConv2d(nn.Module):
    """
    Deformable convolutional layer with configurable
    deformable groups, dilations and groups.
    Code is from:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/misc.py
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            with_modulated_dcn=True,
            kernel_size=3,
            stride=1,
            groups=1,
            dilation=1,
            deformable_groups=1,
            bias=False,
            padding=None
    ):
        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert isinstance(stride, (list, tuple))
            assert isinstance(dilation, (list, tuple))
            assert len(kernel_size) == 2
            assert len(stride) == 2
            assert len(dilation) == 2
            padding = (
                dilation[0] * (kernel_size[0] - 1) // 2,
                dilation[1] * (kernel_size[1] - 1) // 2
            )
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            padding = dilation * (kernel_size - 1) // 2
            offset_base_channels = kernel_size * kernel_size
        if with_modulated_dcn:
            from detectron2.layers.deform_conv import ModulatedDeformConv
            offset_channels = offset_base_channels * 3  # default: 27
            conv_block = ModulatedDeformConv
        else:
            from detectron2.layers.deform_conv import DeformConv
            offset_channels = offset_base_channels * 2  # default: 18
            conv_block = DeformConv
        self.offset = Conv2d(
            in_channels,
            deformable_groups * offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            dilation=dilation
        )
        for l in [self.offset, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            torch.nn.init.constant_(l.bias, 0.)
        self.conv = conv_block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias
        )
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.offset_split = offset_base_channels * deformable_groups * 2

    def forward(self, x, return_offset=False):
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset_mask = self.offset(x)
                x = self.conv(x, offset_mask)
            else:
                offset_mask = self.offset(x)
                offset = offset_mask[:, :self.offset_split, :, :]
                mask = offset_mask[:, self.offset_split:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            if return_offset:
                return x, offset_mask
            return x
        # get output shape
        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride
            )
        ]
        output_shape = [x.shape[0], self.conv.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None
