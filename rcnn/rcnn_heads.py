# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
from typing import Dict, List, Optional, Tuple, Union
import torch
import copy
from torch import nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd.function import Function

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.config import configurable
from detectron2.layers import batched_nms, ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals, add_ground_truth_to_proposals_single_image
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals, select_proposals_with_visible_keypoints, ROIHeads
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.utils.env import TORCH_VERSION
from .mypooler import MyROIPooler
from .my_fast_rcnn_output import MyFastRCNNOutputLayers

__all__ = ["TransformerROIHeads", "CascadeTransformerROIHeads"]


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def add_noise_to_boxes(boxes):
    cxcy_boxes = box_xyxy_to_cxcywh(boxes)
    resize_factor = torch.rand(cxcy_boxes.shape, device=cxcy_boxes.device)
    new_cxcy = cxcy_boxes[..., :2] + cxcy_boxes[..., 2:] * (resize_factor[..., :2] - 0.5) * 0.2
    assert (cxcy_boxes[..., 2:] > 0).all().item()
    new_wh = cxcy_boxes[..., 2:] * (0.8 ** (resize_factor[..., 2:] * 2 - 1))
    assert (new_wh > 0).all().item()
    new_cxcy_boxes = torch.cat([new_cxcy, new_wh], dim=-1)
    new_boxes = box_cxcywh_to_xyxy(new_cxcy_boxes)
    return new_boxes


@ROI_HEADS_REGISTRY.register()
class TransformerROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: MyROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[MyROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[MyROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        add_noise_to_proposals: bool = False,
        encoder_feature: Optional[str] = None,
        random_sample_size: bool = False,
        random_sample_size_upper_bound: float = 1.0,
        random_sample_size_lower_bound: float = 0.8,
        random_proposal_drop: bool = False,
        random_proposal_drop_upper_bound: float = 1.0,
        random_proposal_drop_lower_bound: float = 0.8,
        max_proposal_per_batch: int = 0,
        visualize: bool = False,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes
        self.add_noise_to_proposals = add_noise_to_proposals
        self.encoder_feature = encoder_feature
        self.random_sample_size = random_sample_size
        self.random_proposal_drop = random_proposal_drop
        self.max_proposal_per_batch = max_proposal_per_batch
        self.random_proposal_drop_upper_bound = random_proposal_drop_upper_bound
        self.random_proposal_drop_lower_bound = random_proposal_drop_lower_bound
        self.random_sample_size_upper_bound = random_sample_size_upper_bound
        self.random_sample_size_lower_bound = random_sample_size_lower_bound
        self.visualize = visualize

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["visualize"] = cfg.MODEL.VISUALIZE
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        ret["add_noise_to_proposals"] = cfg.MODEL.ROI_BOX_HEAD.ADD_NOISE_TO_PROPOSALS
        ret["encoder_feature"] = cfg.MODEL.ROI_BOX_HEAD.ENCODER_FEATURE
        ret["random_sample_size"] = cfg.MODEL.ROI_BOX_HEAD.RANDOM_SAMPLE_SIZE
        ret["random_sample_size_upper_bound"] = cfg.MODEL.ROI_BOX_HEAD.RANDOM_SAMPLE_SIZE_UPPER_BOUND
        ret["random_sample_size_lower_bound"] = cfg.MODEL.ROI_BOX_HEAD.RANDOM_SAMPLE_SIZE_LOWER_BOUND
        ret["random_proposal_drop"] = cfg.MODEL.ROI_BOX_HEAD.RANDOM_PROPOSAL_DROP
        ret["random_proposal_drop_upper_bound"] = cfg.MODEL.ROI_BOX_HEAD.RANDOM_PROPOSAL_DROP_UPPER_BOUND
        ret["random_proposal_drop_lower_bound"] = cfg.MODEL.ROI_BOX_HEAD.RANDOM_PROPOSAL_DROP_LOWER_BOUND
        ret["max_proposal_per_batch"] = cfg.MODEL.ROI_BOX_HEAD.MAX_PROPOSAL_PER_BATCH
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        ret["proposal_matcher"] = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = MyROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = MyFastRCNNOutputLayers(cfg, box_head.output_shape)

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        else:
            raise NotImplementedError

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        else:
            raise NotImplementedError

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            losses = self._forward_box(features, proposals, targets)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            if self.visualize:
                pred_instances, attention_maps = self._forward_box(features, proposals)
            else:
                attention_maps = None
                pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            if self.visualize:
                for instance, proposal in zip(pred_instances, proposals):
                    instance._fields["proposal"] = proposal.proposal_boxes.tensor
                for instance, attention in zip(pred_instances, attention_maps):
                    instance._fields["attention"] = attention
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets=None
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = [features[f] for f in self.box_in_features]
        padded_box_features, dec_mask, inds_to_padded_inds = (
            self.box_pooler(box_features, [x.proposal_boxes for x in proposals]))
        enc_feature = None
        enc_mask = None
        if self.box_head.use_encoder_decoder:
            enc_feature = features[self.encoder_feature]
            b = len(proposals)
            h = max([x.image_size[0] for x in proposals])
            w = max([x.image_size[1] for x in proposals])
            enc_mask = torch.ones((b, h, w), dtype=torch.bool, device=padded_box_features.device)
            for c, image_size in enumerate([x.image_size for x in proposals]):
                enc_mask[c, :image_size[0], :image_size[1]] = False
            names = ["res1", "res2", "res3", "res4", "res5"]
            if self.encoder_feature == "p6":
                names.append("p6")
            for name in names:
                if name == "res1":
                    target_shape = ((h+1)//2, (w+1)//2)
                else:
                    x = features[name]
                    target_shape = x.shape[-2:]
                m = enc_mask
                enc_mask = F.interpolate(m[None].float(), size=target_shape).to(torch.bool)[0]

        max_num_proposals = padded_box_features.shape[1]
        normalized_proposals = []
        for x in proposals:
            gt_box = x.proposal_boxes.tensor
            img_h, img_w = x.image_size
            gt_box = gt_box / torch.tensor([img_w, img_h, img_w, img_h],
                                           dtype=torch.float32, device=gt_box.device)
            gt_box = torch.cat([box_xyxy_to_cxcywh(gt_box), gt_box], dim=-1)
            gt_box = F.pad(gt_box, [0, 0, 0, max_num_proposals - gt_box.shape[0]])
            normalized_proposals.append(gt_box)
        normalized_proposals = torch.stack(normalized_proposals, dim=0)

        if self.visualize:
            padded_box_features, attention_maps = self.box_head(enc_feature, enc_mask,
                                                                padded_box_features, dec_mask,
                                                                normalized_proposals)
        else:
            attention_maps = None
            padded_box_features = self.box_head(enc_feature, enc_mask, padded_box_features, dec_mask, normalized_proposals)
        box_features = padded_box_features[inds_to_padded_inds]
        predictions = self.box_predictor(box_features)

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals, targets)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            if self.visualize:
                return pred_instances, attention_maps
            else:
                return pred_instances

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances
        else:
            raise NotImplementedError

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances
        else:
            raise NotImplementedError

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [copy.deepcopy(x.gt_boxes) for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []

        for proposals_per_image, targets_per_image, gt_boxes_per_image in zip(proposals, targets, gt_boxes):
            has_gt = len(targets_per_image) > 0

            if self.add_noise_to_proposals:
                proposals_per_image.proposal_boxes.tensor = (
                    add_noise_to_boxes(proposals_per_image.proposal_boxes.tensor))

            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            if not torch.any(matched_labels == 1) and self.proposal_append_gt:
                gt_boxes_per_image.tensor = add_noise_to_boxes(gt_boxes_per_image.tensor)
                proposals_per_image = add_ground_truth_to_proposals_single_image(gt_boxes_per_image,
                                                                                 proposals_per_image)

                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes)

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
                proposals_per_image.set('gt_idxs', sampled_targets)
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
                proposals_per_image.set('gt_idxs', torch.zeros_like(sampled_idxs))

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.
        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.
        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        if self.random_sample_size:
            diff = self.random_sample_size_upper_bound - self.random_sample_size_lower_bound
            sample_factor = self.random_sample_size_upper_bound - np.random.rand(1)[0] * diff
            nms_topk = int(matched_idxs.shape[0] * sample_factor)
            matched_idxs = matched_idxs[:nms_topk]
            matched_labels = matched_labels[:nms_topk]

        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        if self.random_proposal_drop:
            diff = self.random_proposal_drop_upper_bound - self.random_proposal_drop_lower_bound
            sample_factor = self.random_proposal_drop_upper_bound - np.random.rand(1)[0] * diff
            nms_topk = int(sampled_idxs.shape[0] * sample_factor)
            subsample_idxs = np.random.choice(sampled_idxs.shape[0], nms_topk, replace=False)
            subsample_idxs = torch.from_numpy(subsample_idxs).to(sampled_idxs.device)
            sampled_idxs = sampled_idxs[subsample_idxs]

        return sampled_idxs, gt_classes[sampled_idxs]


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


@ROI_HEADS_REGISTRY.register()
class CascadeTransformerROIHeads(TransformerROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    """
    Shengcao: Please complete the following initialization functions with proper variable names for TSP-RCNN.
    Later functions will use:
        num_cascade_stages: int
        box_head: List[nn.Module], length num_cascade_stages
        box_predictor: List[nn.Module], length num_cascade_stages
        proposal_matchers: List[Matcher], length num_cascade_stages

    Adapted from:
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/roi_heads/cascade_rcnn.py
    """

    @configurable
    def __init__(
        self,
        *,
        inherit_match: bool,
        box_in_features: List[str],
        box_pooler: MyROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        **kwargs):
        assert "proposal_matcher" not in kwargs, (
            "CascadeROIHeads takes 'proposal_matchers=' for each stage instead "
            "of one 'proposal_matcher='."
        )
        # The first matcher matches RPN proposals with ground truth, done in the base class
        kwargs["proposal_matcher"] = proposal_matchers[0]
        num_stages = self.num_cascade_stages = len(box_heads)
        box_heads = nn.ModuleList(box_heads)
        box_predictors = nn.ModuleList(box_predictors)
        assert len(box_predictors) == num_stages, f"{len(box_predictors)} != {num_stages}!"
        assert len(proposal_matchers) == num_stages, f"{len(proposal_matchers)} != {num_stages}!"
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_heads,
            box_predictor=box_predictors,
            **kwargs,
        )
        self.proposal_matchers = proposal_matchers
        self.inherit_match = inherit_match

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["inherit_match"] = cfg.MODEL.ROI_BOX_CASCADE_HEAD.INHERIT_MATCH
        ret.pop("proposal_matcher")
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # cascade-specific
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        assert len(cascade_bbox_reg_weights) == len(cascade_ious)
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = MyROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        pooled_shape = ShapeSpec(
            channels=in_channels, height=pooler_resolution, width=pooler_resolution
        )

        box_heads, box_predictors, proposal_matchers = [], [], []
        share_output_head = cfg.MODEL.ROI_BOX_CASCADE_HEAD.SHARE_OUTPUT_HEAD
        fine_tune_head = cfg.MODEL.ROI_BOX_CASCADE_HEAD.FINE_TUNE_HEAD
        box_predictor = None
        for match_iou, bbox_reg_weights in zip(cascade_ious, cascade_bbox_reg_weights):
            box_head = build_box_head(cfg, pooled_shape)
            box_heads.append(box_head)
            if box_predictor is None or (not share_output_head and not fine_tune_head):
                box_predictor = MyFastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(
                        weights=bbox_reg_weights),
                )
            elif fine_tune_head:
                box_predictor = MyFastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(
                        weights=bbox_reg_weights),
                    finetune_on_set=True,
                )
            box_predictors.append(box_predictor)
            proposal_matchers.append(Matcher([match_iou], [0, 1], allow_low_quality_matches=False))

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_heads": box_heads,
            "box_predictors": box_predictors,
            "proposal_matchers": proposal_matchers,
        }

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets=None
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        prev_box_features = None
        prev_proposals = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are used to create the input
                # proposals of the next stage.
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if self.training:
                    if self.inherit_match:
                        proposals = self._inherit_labels(proposals, prev_proposals)
                    else:
                        proposals = self._match_and_label_boxes(proposals, k, targets)
            prev_box_features, predictions = self._run_stage(box_features, prev_box_features, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            prev_proposals = proposals
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = predictor.losses(predictions, proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            if self.train_on_pred_boxes:
                # Default False
                raise NotImplementedError
            return losses
        else:
            # Use the boxes of the last head
            predictor, predictions, proposals = head_outputs[-1]
            pred_instances, _ = predictor.inference(predictions, proposals)
            return pred_instances

    @torch.no_grad()
    def _inherit_labels(self, proposals, prev_proposals):
        for proposals_per_image, prev_proposals_per_image in zip(proposals, prev_proposals):
            proposals_per_image.gt_classes = prev_proposals_per_image.gt_classes
            proposals_per_image.gt_boxes = prev_proposals_per_image.gt_boxes
            proposals_per_image.gt_idxs = prev_proposals_per_image.gt_idxs

        return proposals

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        """
        Match proposals with ground-truth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.
        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances
        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)

            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
                gt_idxs = matched_idxs
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
                gt_idxs = torch.zeros_like(matched_idxs)

            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes
            proposals_per_image.gt_idxs = gt_idxs

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    def _run_stage(self, box_features, prev_box_features, proposals, stage):
        """
        Args:
            box_features (list[Tensor]): #lvl input features to ROIHeads
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage
        Returns:
            Same output as `FastRCNNOutputLayers.forward()`.
        """
        padded_box_features, dec_mask, inds_to_padded_inds = (
            self.box_pooler(box_features, [x.proposal_boxes for x in proposals]))
        enc_feature = None
        enc_mask = None
        if self.box_head[stage].use_encoder_decoder:
            # Default False
            raise NotImplementedError

        max_num_proposals = padded_box_features.shape[1]
        normalized_proposals = []
        for x in proposals:
            gt_box = x.proposal_boxes.tensor
            img_h, img_w = x.image_size
            gt_box = gt_box / torch.tensor([img_w, img_h, img_w, img_h],
                                           dtype=torch.float32, device=gt_box.device)
            gt_box = torch.cat([box_xyxy_to_cxcywh(gt_box), gt_box], dim=-1)
            gt_box = F.pad(gt_box, [0, 0, 0, max_num_proposals - gt_box.shape[0]])
            normalized_proposals.append(gt_box)
        normalized_proposals = torch.stack(normalized_proposals, dim=0)

        # Shengcao: Not sure if this gradient scaling is necessary, I just copy it here

        # The original implementation averages the losses among heads,
        # but scale up the parameter gradients of the heads.
        # This is equivalent to adding the losses among heads,
        # but scale down the gradients on features.
        padded_box_features = _ScaleGradient.apply(padded_box_features, 1.0 / self.num_cascade_stages)

        padded_box_features = self.box_head[stage](enc_feature, enc_mask, padded_box_features, dec_mask, normalized_proposals, prev_box_features)
        box_features = padded_box_features[inds_to_padded_inds]

        return padded_box_features, self.box_predictor[stage](box_features)

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)
        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals
