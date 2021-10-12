# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

from .focal_loss import focal_loss_weight
from .soft_nms import batched_soft_nms
from .matcher import generalized_box_iou, HungarianMatcher

__all__ = ["MyFastRCNNOutputLayers"]

logger = logging.getLogger(__name__)


def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image,
                        soft_nms_enabled, soft_nms_method, soft_nms_sigma, soft_nms_prune):
    """
    Call `fast_rcnn_inference_single_image` for all images.
    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.
    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image,
            soft_nms_enabled, soft_nms_method, soft_nms_sigma, soft_nms_prune
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image,
    soft_nms_enabled, soft_nms_method, soft_nms_sigma, soft_nms_prune
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).
    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.
    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    if not soft_nms_enabled:
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    else:
        keep, soft_nms_scores = batched_soft_nms(
            boxes,
            scores,
            filter_inds[:, 1],
            soft_nms_method,
            soft_nms_sigma,
            nms_thresh,
            soft_nms_prune,
            topk_per_image,
        )
        scores[keep] = soft_nms_scores
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def detr_fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image,
                             soft_nms_enabled, soft_nms_method, soft_nms_sigma, soft_nms_prune):
    """
    Call `fast_rcnn_inference_single_image` for all images.
    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.
    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        detr_fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image,
            soft_nms_enabled, soft_nms_method, soft_nms_sigma, soft_nms_prune
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def detr_fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image,
    soft_nms_enabled, soft_nms_method, soft_nms_sigma, soft_nms_prune
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).
    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.
    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = (scores > score_thresh) & (scores > torch.max(scores, dim=-1, keepdim=True)[0] * 0.75)
    # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    if not soft_nms_enabled:
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    else:
        keep, soft_nms_scores = batched_soft_nms(
            boxes,
            scores,
            filter_inds[:, 1],
            soft_nms_method,
            soft_nms_sigma,
            nms_thresh,
            soft_nms_prune,
        )
        scores[keep] = soft_nms_scores
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


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


class MyFastRCNNOutputs(object):
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_obj_logits,
        pred_proposal_deltas,
        proposals,
        targets=None,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        l1_weight=1.0,
        giou_weight=2.0,
        eos_weight=1.0,
        use_obj_loss=False,
        finetune_on_set=False,
        cls_head_no_bg=False,
        separate_obj_cls=False,
        use_detr_loss=False,
        matcher=None,
        num_classes=None,
        empty_weight=None,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_obj_logits = pred_obj_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.eos_weight = eos_weight
        self.use_obj_loss = use_obj_loss
        self.finetune_on_set = finetune_on_set
        self.cls_head_no_bg = cls_head_no_bg
        self.separate_obj_cls = separate_obj_cls
        self.use_detr_loss = use_detr_loss
        self.matcher = matcher
        self.num_classes = num_classes
        self.empty_weight = empty_weight

        self.image_shapes = [x.image_size for x in proposals]
        # print("image_shapes", self.image_shapes)
        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            self.proposals_per_image = [p.proposal_boxes for p in proposals]
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                self.gt_boxes_per_image = [p.gt_boxes for p in proposals]
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
                self.gt_classes_per_image = [p.gt_classes for p in proposals]
                self.gt_idxs_per_image = [p.gt_idxs for p in proposals]
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0  # no instances found
        self.targets = targets

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self, true_fg_inds=None):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            loss_weight = torch.ones(self.pred_class_logits.shape[0], device=self.pred_class_logits.device)
            bg_inds = (self.gt_classes == self.pred_class_logits.shape[1] - 1)
            loss_weight[bg_inds] = self.eos_weight
            loss_normalizer = loss_weight.sum() + 1e-6
            if self.cls_head_no_bg:
                loss_weight[bg_inds] = 0.0
            if true_fg_inds is not None:
                fg_inds = nonzero_tuple(true_fg_inds)[0]
                if len(fg_inds) == 0:
                    return 0.0 * self.pred_class_logits.sum()
                ce_loss = F.cross_entropy(self.pred_class_logits[fg_inds], self.gt_classes[fg_inds], reduction="none")
                ce_loss = (ce_loss * loss_weight[fg_inds]).sum() / loss_normalizer
            else:
                ce_loss = F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="none")
                ce_loss = (ce_loss * loss_weight).sum() / loss_normalizer
            return ce_loss

    def box_reg_loss(self, true_fg_inds=None):
        """
        Compute the smooth L1 loss for box regression.
        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        if true_fg_inds is not None:
            fg_inds = nonzero_tuple(true_fg_inds)[0]
            if len(fg_inds) == 0:
                return 0.0 * self.pred_proposal_deltas.sum()
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        elif self.box_reg_loss_type == "smooth_l1+giou":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg_l1 = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
            loss_box_reg_giou = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
            loss_box_reg = self.l1_weight * loss_box_reg_l1 + self.giou_weight * loss_box_reg_giou
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        if true_fg_inds is not None:
            loss_box_reg = loss_box_reg * 3.0
        return loss_box_reg

    def obj_loss(self):
        if self._no_instances:
            if self.separate_obj_cls:
                return 0.0 * self.pred_obj_logits.sum(), None
            else:
                return 0.0 * self.pred_class_logits.sum(), None

        device = self.pred_proposal_deltas.device
        pred_class_logits_per_image = self.pred_class_logits.split(self.num_preds_per_image, 0)
        pred_obj_logits_per_image = None
        if self.separate_obj_cls:
            pred_obj_logits_per_image = self.pred_obj_logits.split(self.num_preds_per_image, 0)

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        all_inds = nonzero_tuple(self.gt_classes <= bg_class_ind)[0]
        if cls_agnostic_bbox_reg:
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[all_inds]
            fg_gt_classes[fg_gt_classes == bg_class_ind] = bg_class_ind - 1
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        if self.use_detr_loss:
            pred_boxes_per_image = self._predict_boxes().split(self.num_preds_per_image, 0)

            num_gt_boxes = 2.0 * len(self.num_preds_per_image)
            total_pred_logits = []
            total_target_classes = []

            loss_giou = 0.0
            for i in range(len(self.num_preds_per_image)):
                pred_logits = pred_class_logits_per_image[i].unsqueeze(0)
                pred_boxes = pred_boxes_per_image[i].unsqueeze(0)
                target = self.targets[i]
                num_gt_boxes += target.gt_classes.shape[0]

                outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
                targets = [{"boxes": target.gt_boxes.tensor, "labels": target.gt_classes}]

                matched_indices = self.matcher(outputs, targets, only_gious=True)

                idx = self._get_src_permutation_idx(matched_indices)
                target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, matched_indices)])
                target_classes = torch.full(pred_logits.shape[:2], self.num_classes,
                                            dtype=torch.int64, device=pred_logits.device)
                target_classes[idx] = target_classes_o
                total_pred_logits.append(pred_logits.squeeze(0))
                total_target_classes.append(target_classes.squeeze(0))

                src_boxes = pred_boxes[idx]
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, matched_indices)], dim=0)
                loss_giou_per_image = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
                loss_giou = loss_giou + loss_giou_per_image.sum()

            total_pred_logits = torch.cat(total_pred_logits, dim=0)
            total_target_classes = torch.cat(total_target_classes, dim=0)

            if all_inds.numel() > 0:
                loss_ce = F.cross_entropy(total_pred_logits, total_target_classes, self.empty_weight)
                loss_giou = loss_giou / num_gt_boxes
            else:
                loss_ce = total_pred_logits.sum() * 0
                loss_giou = pred_boxes_per_image[0].sum() * 0

            ret_losses = {
                "loss_obj": loss_ce * 0.5,
                "loss_box_reg": loss_giou * 2.0,
            }
            return ret_losses
        else:
            with torch.no_grad():
                all_cost_giou = giou_loss(
                    self._predict_boxes()[all_inds[:, None], gt_class_cols],
                    self.gt_boxes.tensor[all_inds],
                    reduction="none",
                )
                cost_giou_per_image = all_cost_giou.split(self.num_preds_per_image, 0)

            adjusted_targets = []
            max_gt_idxs = []
            for i in range(len(self.num_preds_per_image)):
                num_preds = self.num_preds_per_image[i]
                pred_class_logits = pred_class_logits_per_image[i]
                gt_classes = self.gt_classes_per_image[i]
                gt_idxs = self.gt_idxs_per_image[i]
                cost_giou = cost_giou_per_image[i]
                if self.separate_obj_cls:
                    pred_obj_logits = pred_obj_logits_per_image[i]
                    cost_class = - (pred_class_logits.softmax(-1)[range(num_preds), gt_classes]
                                    * pred_obj_logits.softmax(-1)[..., 0])
                else:
                    cost_class = - pred_class_logits.softmax(-1)[range(num_preds), gt_classes]

                # sorted by proposals
                total_cost = cost_class + self.giou_weight * cost_giou

                fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
                if self.separate_obj_cls:
                    adjusted_target = torch.full((num_preds,), 1, device=device, dtype=torch.int64)
                else:
                    adjusted_target = torch.full((num_preds,), bg_class_ind, device=device, dtype=torch.int64)

                if len(gt_idxs):
                    cost = torch.full((gt_idxs.max().item() + 1, num_preds), 1e9, device=device, dtype=torch.float32)
                    range_idxs = torch.arange(start=0, end=num_preds, device=device, dtype=torch.int64)
                    cost[gt_idxs[fg_inds], range_idxs[fg_inds]] = total_cost[fg_inds]

                    matched_values, matched_proposal_idxs = torch.min(cost, dim=-1)
                    valid_proposal = matched_values < 9e8
                    matched_proposal_idxs = matched_proposal_idxs[valid_proposal]
                    if len(matched_proposal_idxs):
                        if self.separate_obj_cls:
                            adjusted_target[matched_proposal_idxs] = torch.zeros_like(gt_classes[matched_proposal_idxs])
                        else:
                            adjusted_target[matched_proposal_idxs] = gt_classes[matched_proposal_idxs]
                    max_gt_idxs.append(gt_idxs.max().item() + 1)
                else:
                    max_gt_idxs.append(0)
                adjusted_targets.append(adjusted_target)

            num_instances = self.gt_classes.numel()
            fg_num_accurate = None
            num_false_negative = None
            adjusted_targets = torch.cat(adjusted_targets, dim=0)
            if self.separate_obj_cls:
                pred_objs = self.pred_obj_logits.argmax(dim=1)
                true_fg_inds = (adjusted_targets == 0)
                if len(nonzero_tuple(true_fg_inds)[0]) == 0:
                    return 0.0 * self.pred_obj_logits.sum(), true_fg_inds

                num_fg = true_fg_inds.nonzero().numel()
                num_accurate = (pred_objs == adjusted_targets).nonzero().numel()
            else:
                pred_classes = self.pred_class_logits.argmax(dim=1)
                bg_class_ind = self.pred_class_logits.shape[1] - 1
                true_fg_inds = (adjusted_targets >= 0) & (adjusted_targets < bg_class_ind)
                if len(nonzero_tuple(true_fg_inds)[0]) == 0:
                    return 0.0 * self.pred_class_logits.sum(), true_fg_inds

                num_fg = true_fg_inds.nonzero().numel()
                fg_gt_classes = adjusted_targets[true_fg_inds]
                fg_pred_classes = pred_classes[true_fg_inds]

                num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
                num_accurate = (pred_classes == adjusted_targets).nonzero().numel()
                fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

            empty_weight = torch.ones(self.gt_classes.shape[0], device=self.pred_class_logits.device)
            empty_weight[~true_fg_inds] = self.eos_weight
            loss_normalizer = empty_weight.sum() + 1e-6

            if self.separate_obj_cls:
                ret_loss = F.cross_entropy(self.pred_obj_logits, adjusted_targets, reduction="none")
                focal_weight = focal_loss_weight(self.pred_obj_logits, adjusted_targets)
            else:
                ret_loss = F.cross_entropy(self.pred_class_logits, adjusted_targets, reduction="none")
                focal_weight = focal_loss_weight(self.pred_class_logits, adjusted_targets)
            ret_loss = (ret_loss * focal_weight * empty_weight).sum() / loss_normalizer

            storage = get_event_storage()
            storage.put_scalar("fast_rcnn/num_fg_v2", num_fg / len(self.num_preds_per_image))
            storage.put_scalar("fast_rcnn/num_fg", sum(max_gt_idxs) / len(self.num_preds_per_image))
            if num_instances > 0:
                storage.put_scalar("fast_rcnn/obj_cls_accuracy", num_accurate / num_instances)
                if not self.separate_obj_cls:
                    if num_fg > 0:
                        storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                        storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

            return ret_loss, true_fg_inds

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.
        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        true_fg_inds = None
        ret = dict()
        if self.separate_obj_cls:
            if not self.use_obj_loss:
                raise NotImplementedError
            ret["loss_obj"], true_fg_inds = self.obj_loss()
            if self.finetune_on_set:
                ret["loss_box_reg"] = self.box_reg_loss(true_fg_inds)
                ret["loss_cls"] = self.softmax_cross_entropy_loss(true_fg_inds)
            else:
                ret["loss_box_reg"] = self.box_reg_loss()
                ret["loss_cls"] = self.softmax_cross_entropy_loss()
        else:
            if self.use_detr_loss:
                ret = self.obj_loss()
            else:
                if self.use_obj_loss:
                    ret["loss_obj"], true_fg_inds = self.obj_loss()
                else:
                    ret["loss_cls"] = self.softmax_cross_entropy_loss()
                if self.finetune_on_set:
                    ret["loss_box_reg"] = self.box_reg_loss(true_fg_inds)
                else:
                    ret["loss_box_reg"] = self.box_reg_loss()
        return ret


class MyFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        soft_nms_enabled=False,
        soft_nms_method="gaussian",
        soft_nms_sigma=0.5,
        soft_nms_prune=0.001,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        use_obj_loss: bool = False,
        eos_weight: float = 1.0,
        l1_weight: float = 1.0,
        giou_weight: float = 2.0,
        finetune_on_set: bool = False,
        cls_head_no_bg: bool = False,
        detr_eval_protocol: bool = False,
        separate_obj_cls: bool = False,
        use_detr_loss: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    "loss_cls" - applied to classification loss
                    "loss_box_reg" - applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = Linear(input_size, num_classes + 1)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        # self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)
        self.bbox_pred = MLP(input_size, input_size, num_bbox_reg_classes * box_dim, 3)

        self.obj_score = None
        if use_obj_loss and separate_obj_cls:
            self.obj_score = Linear(input_size, 2)
            nn.init.normal_(self.obj_score.weight, std=0.01)
            nn.init.constant_(self.obj_score.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.soft_nms_enabled = soft_nms_enabled
        self.soft_nms_method = soft_nms_method
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_prune = soft_nms_prune
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight
        self.use_obj_loss = use_obj_loss
        self.eos_weight = eos_weight
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.finetune_on_set = finetune_on_set
        self.cls_head_no_bg = cls_head_no_bg
        self.detr_eval_protocol = detr_eval_protocol
        self.separate_obj_cls = separate_obj_cls
        self.use_detr_loss = use_detr_loss
        if use_detr_loss:
            empty_weight = torch.ones(num_classes + 1)
            empty_weight[-1] = self.eos_weight
            self.register_buffer('empty_weight', empty_weight)
            self.num_classes = num_classes
            self.matcher = HungarianMatcher(1, 2, 5)
        else:
            self.empty_weight = None
            self.num_classes = None
            self.matcher = None

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "soft_nms_enabled": cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED,
            "soft_nms_method": cfg.MODEL.ROI_HEADS.SOFT_NMS_METHOD,
            "soft_nms_sigma": cfg.MODEL.ROI_HEADS.SOFT_NMS_SIGMA,
            "soft_nms_prune": cfg.MODEL.ROI_HEADS.SOFT_NMS_PRUNE,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "l1_weight": cfg.MODEL.ROI_BOX_HEAD.L1_WEIGHT,
            "giou_weight": cfg.MODEL.ROI_BOX_HEAD.GIOU_WEIGHT,
            "loss_weight": {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            "use_obj_loss": cfg.MODEL.ROI_BOX_HEAD.USE_OBJ_LOSS,
            "eos_weight": cfg.MODEL.ROI_BOX_HEAD.EOS_COEF,
            "finetune_on_set": cfg.MODEL.ROI_BOX_HEAD.FINETUNE_ON_SET,
            "cls_head_no_bg": cfg.MODEL.ROI_BOX_HEAD.CLS_HEAD_NO_BG,
            "detr_eval_protocol": cfg.MODEL.ROI_BOX_HEAD.DETR_EVAL_PROTOCOL,
            "separate_obj_cls": cfg.MODEL.ROI_BOX_HEAD.SEPARATE_OBJ_CLS,
            "use_detr_loss": cfg.MODEL.ROI_BOX_HEAD.USE_DETR_LOSS,
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.
        Returns:
            Tensor: shape (N,K+1), scores for each of the N box. Each row contains the scores for
                K object categories and 1 background class.
            Tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4), or (N,4)
                for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        obj_scores = None
        if self.obj_score is not None:
            obj_scores = self.obj_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, obj_scores, proposal_deltas

    # TODO: move the implementation to this class.
    def losses(self, predictions, proposals, targets=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, obj_scores, proposal_deltas = predictions
        losses = MyFastRCNNOutputs(
            self.box2box_transform,
            scores,
            obj_scores,
            proposal_deltas,
            proposals,
            targets,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            self.l1_weight,
            self.giou_weight,
            self.eos_weight,
            self.use_obj_loss,
            self.finetune_on_set,
            self.cls_head_no_bg,
            self.separate_obj_cls,
            self.use_detr_loss,
            self.matcher,
            self.num_classes,
            self.empty_weight,
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def inference(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        if self.detr_eval_protocol:
            return detr_fast_rcnn_inference(
                boxes,
                scores,
                image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_topk_per_image,
                self.soft_nms_enabled,
                self.soft_nms_method,
                self.soft_nms_sigma,
                self.soft_nms_prune,
            )
        else:
            return fast_rcnn_inference(
                boxes,
                scores,
                image_shapes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_topk_per_image,
                self.soft_nms_enabled,
                self.soft_nms_method,
                self.soft_nms_sigma,
                self.soft_nms_prune,
            )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, obj_scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, obj_scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        if obj_scores is None:
            new_probs = probs
        else:
            obj_probs = F.softmax(obj_scores, dim=-1)
            new_probs = torch.cat([probs[..., :-1] * obj_probs[..., :-1],
                                   probs[..., -1:] * obj_probs[..., -1:]], dim=-1)
        return new_probs.split(num_inst_per_image, dim=0)
