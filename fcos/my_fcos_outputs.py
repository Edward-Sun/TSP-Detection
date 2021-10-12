import logging
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes
from detectron2.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit

from detectron2.layers import batched_nms
from detectron2.utils.events import get_event_storage
from .matcher import generalized_box_iou, HungarianMatcher

logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:
    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization
Naming convention:
    labels: refers to the ground-truth class of an position.
    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.
    logits_pred: predicted classification scores in [-inf, +inf];

    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 
    ctrness_pred: predicted centerness scores
"""


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def ml_nms(boxlist, nms_thresh, max_proposals=-1,
           score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Args:
        boxlist (detectron2.structures.Boxes):
        nms_thresh (float):
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str):
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist


class IOULoss(nn.Module):
    """
    Intersetion Over Union (IoU) loss which supports three
    different IoU computations:
    * IoU
    * Linear IoU
    * gIoU
    """
    def __init__(self, loc_loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None, sum=True):
        """
        Args:
            pred: Nx4 predicted bounding boxes
            target: Nx4 target bounding boxes
            weight: N loss weight for each instance
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                        torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if sum:
            if weight is not None:
                return (losses * weight).sum()
            else:
                return losses.sum()
        else:
            return losses


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
              (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)


class FCOSOutputs(nn.Module):
    def __init__(self, cfg):
        super(FCOSOutputs, self).__init__()

        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
        self.center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.radius = cfg.MODEL.FCOS.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
        self.pre_nms_topk_train = cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN
        self.post_nms_topk_train = cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN
        self.loc_loss_func = IOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)

        self.pre_nms_thresh_test = cfg.MODEL.FCOS.INFERENCE_TH_TEST
        self.pre_nms_topk_test = cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST
        self.post_nms_topk_test = cfg.MODEL.FCOS.POST_NMS_TOPK_TEST
        self.nms_thresh = cfg.MODEL.FCOS.NMS_TH
        self.thresh_with_ctr = cfg.MODEL.FCOS.THRESH_WITH_CTR

        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.strides = cfg.MODEL.FCOS.FPN_STRIDES

        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi
        self.use_obj_loss = cfg.MODEL.FCOS.USE_OBJ_LOSS
        self.use_detr_loss = cfg.MODEL.FCOS.USE_DETR_LOSS
        self.giou_weight = cfg.MODEL.FCOS.GIOU_WEIGHT
        self.predict_without_ctr = cfg.MODEL.FCOS.PREDICT_WITHOUT_CTR
        self.eos_weight = cfg.MODEL.FCOS.EOS_COEF
        self.only_reweight_fg = cfg.MODEL.FCOS.ONLY_REWEIGHT_FG
        self.class_denorm_type = cfg.MODEL.FCOS.CLASS_DENORM_TYPE
        if self.use_detr_loss:
            self.matcher = HungarianMatcher(1, 2, 5)
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_weight
            self.register_buffer('empty_weight', empty_weight)

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def get_ground_truth(self, locations, gt_instances):
        num_loc_list = [len(loc) for loc in locations]

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        training_targets = self.compute_targets_for_locations(
            locations, gt_instances, loc_to_size_range, num_loc_list
        )

        training_targets["locations"] = [locations.clone() for _ in range(len(gt_instances))]
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i for i in range(len(gt_instances))
        ]

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(training_targets["locations"])
        ]

        # we normalize reg_targets by FPN's strides here
        reg_targets = training_targets["reg_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])

        return training_targets

    def get_sample_region(self, boxes, strides, num_loc_list, loc_xs, loc_ys, bitmasks=None, radius=1):
        if bitmasks is not None:
            _, h, w = bitmasks.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges, num_loc_list):
        labels = []
        reg_targets = []
        target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        # loop for each image
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                if targets_per_im.has("gt_bitmasks_full"):
                    bitmasks = targets_per_im.gt_bitmasks_full
                else:
                    bitmasks = None
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, num_loc_list, xs, ys,
                    bitmasks=bitmasks, radius=self.radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            target_inds.append(target_inds_per_im)

        return {
            "labels": labels,
            "reg_targets": reg_targets,
            "target_inds": target_inds
        }

    def losses(self, logits_pred, reg_pred, ctrness_pred, locations, gt_instances,
               top_feats=None, local_logits_pred=None, poi_idx_flatten=None, batch_size=None):
        """
        Return the losses from a set of FCOS predictions and their associated ground-truth.
        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        training_targets = self.get_ground_truth(locations, gt_instances)

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.

        N = local_logits_pred[0].shape[0]

        instances = Instances((0, 0))
        instances.labels = cat([
            # Reshape: (N*Hi*Wi,) -> (N*Hi*Wi,)
            x.reshape(N, -1) for x in training_targets["labels"]
        ], dim=1).view(-1)
        instances.reg_targets = cat([
            # Reshape: (N*Hi*Wi, 4) -> (N*Hi*Wi, 4)
            x.reshape(N, -1, 4) for x in training_targets["reg_targets"]
        ], dim=1).view(-1, 4)
        instances.target_inds = cat([
            x.reshape(N, -1) for x in training_targets["target_inds"]
        ], dim=1).view(-1)

        # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
        ctrness_pred = ctrness_pred.view(-1)
        instances.local_logits_pred = cat([
            # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
            x.permute(0, 2, 3, 1).reshape(N, -1, self.num_classes) for x in local_logits_pred
        ], dim=1).view(-1, self.num_classes)

        return self.fcos_losses(instances, logits_pred, reg_pred, ctrness_pred, poi_idx_flatten,
                                locations, gt_instances, batch_size)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def fcos_losses(self, instances, logits_pred, reg_pred, ctrness_pred, poi_idx_flatten,
                    locations=None, gt_instances=None, batch_size=None):
        num_classes = instances.local_logits_pred.size(1)
        assert num_classes == self.num_classes

        labels = instances.labels.flatten()

        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        class_target = torch.zeros_like(instances.local_logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        local_class_loss = sigmoid_focal_loss_jit(
            instances.local_logits_pred,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg

        # Start Transformer Set-Prediction Loss
        transformer_labels = labels[poi_idx_flatten]
        transformer_pos_inds = torch.nonzero(transformer_labels != num_classes).squeeze(1)
        transformer_num_pos_local = transformer_pos_inds.numel()
        num_gpus = get_world_size()
        transformer_total_num_pos = reduce_sum(transformer_pos_inds.new_tensor([transformer_num_pos_local])).item()
        transformer_num_pos_avg = max(transformer_total_num_pos / num_gpus, 1.0)

        if not self.use_detr_loss:
            reg_pred = reg_pred[transformer_pos_inds]
        ctrness_pred = ctrness_pred[transformer_pos_inds]
        instances = instances[poi_idx_flatten][transformer_pos_inds]

        ctrness_targets = compute_ctrness_targets(instances.reg_targets)
        ctrness_targets_sum = ctrness_targets.sum()
        loss_denorm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)
        instances.gt_ctrs = ctrness_targets

        acc_foreground_avg = None
        acc_background_avg = None
        acc_set_prediction_fg_avg = None
        acc_set_prediction_bg_avg = None
        class_loss_denorm = None
        transformer_num_set_pos_avg = None
        if self.use_detr_loss:
            targets = []
            for instance in gt_instances:
                targets.append({"boxes": instance.gt_boxes.tensor, "labels": instance.gt_classes})

            tmp_locations = []
            for stride, location in zip(self.strides, locations):
                tmp_locations.append(torch.cat([torch.ones_like(location[:, :1]) * stride, location], dim=1))
            locations = torch.cat(tmp_locations, dim=0).view(1, -1, 3).repeat(batch_size, 1, 1).view(-1, 3)
            locations = locations[poi_idx_flatten]

            # left, top, right, bottom
            x1 = locations[:, 1] - reg_pred[:, 0] * locations[:, 0]
            y1 = locations[:, 2] - reg_pred[:, 1] * locations[:, 0]
            x2 = locations[:, 1] + reg_pred[:, 2] * locations[:, 0]
            y2 = locations[:, 2] + reg_pred[:, 3] * locations[:, 0]

            # per_locations[:, 0] - per_box_regression[:, 0],
            # per_locations[:, 1] - per_box_regression[:, 1],
            # per_locations[:, 0] + per_box_regression[:, 2],
            # per_locations[:, 1] + per_box_regression[:, 3],

            pred_logits = torch.cat([logits_pred, torch.zeros_like(logits_pred)[:, :1]], dim=1)
            pred_boxes = torch.stack([x1, y1, x2, y2], dim=1).view(batch_size, -1, 4)
            pred_logits = pred_logits.view(batch_size, pred_boxes.shape[1], -1)
            outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

            matched_indices = self.matcher(outputs, targets, only_gious=True)

            if instances.target_inds.numel() > 0:
                idx = self._get_src_permutation_idx(matched_indices)
                target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, matched_indices)])
                target_classes = torch.full(pred_logits.shape[:2], self.num_classes,
                                            dtype=torch.int64, device=pred_logits.device)
                target_classes[idx] = target_classes_o
                loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)

                src_boxes = pred_boxes[idx]
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, matched_indices)], dim=0)
                loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
                loss_giou = loss_giou.sum() / transformer_num_pos_avg

                ctrness_loss = F.binary_cross_entropy_with_logits(
                    ctrness_pred,
                    ctrness_targets,
                    reduction="sum"
                ) / transformer_num_pos_avg
            else:
                loss_ce = pred_logits.sum() * 0
                loss_giou = pred_boxes.sum() * 0
                ctrness_loss = ctrness_pred.sum() * 0
            losses = {
                "loss_fcos_local_cls": local_class_loss,
                "loss_fcos_cls": loss_ce,
                "loss_fcos_loc": loss_giou * 10.0,
                "loss_fcos_ctr": ctrness_loss
            }
            extras = {}
        else:
            if self.use_obj_loss:
                gt_idxs = instances.target_inds
                gt_classes = transformer_labels
                fg_inds = (gt_classes != num_classes)
                num_preds = logits_pred.shape[0]

                class_target = torch.full_like(gt_classes, self.num_classes)
                matched_proposal_idxs = None
                if gt_idxs.numel() > 0:
                    cost_class = - logits_pred.sigmoid()[fg_inds]
                    cost_class = cost_class[range(transformer_pos_inds.shape[0]), gt_classes[fg_inds]]
                    # cost_class is negative
                    cost_class = cost_class * (ctrness_targets + 1e-6)
                    cost_giou = self.loc_loss_func(reg_pred, instances.reg_targets, sum=False)
                    total_cost = cost_class + self.giou_weight * cost_giou

                    device = total_cost.device
                    cost = torch.full((gt_idxs.max().item() + 1, num_preds), 1e9, device=device, dtype=torch.float32)
                    range_idxs = torch.arange(start=0, end=num_preds, device=device, dtype=torch.int64)
                    cost[gt_idxs, range_idxs[fg_inds]] = total_cost

                    matched_values, matched_proposal_idxs = torch.min(cost, dim=-1)
                    valid_proposal = matched_values < 9e8
                    matched_proposal_idxs = matched_proposal_idxs[valid_proposal]
                    class_target[matched_proposal_idxs] = gt_classes[matched_proposal_idxs]

                if matched_proposal_idxs is None:
                    transformer_num_set_pos_local = 0.0
                else:
                    transformer_num_set_pos_local = matched_proposal_idxs.numel()
                num_gpus = get_world_size()
                transformer_total_num_set_pos = reduce_sum(
                    transformer_pos_inds.new_tensor([transformer_num_set_pos_local])).item()
                transformer_num_set_pos_avg = max(transformer_total_num_set_pos / num_gpus, 1.0)

                empty_weight = torch.ones_like(logits_pred)
                if self.only_reweight_fg:
                    false_positive_inds = fg_inds & (class_target == num_classes)
                    empty_weight[false_positive_inds, transformer_labels[false_positive_inds]] = self.eos_weight
                else:
                    empty_weight[fg_inds] = torch.ones_like(empty_weight[fg_inds]) * self.eos_weight
                class_loss_denorm = transformer_num_set_pos_avg + self.eos_weight * transformer_num_pos_avg

                focal_loss_class_target = torch.zeros_like(logits_pred)
                focal_loss_class_target[matched_proposal_idxs, transformer_labels[matched_proposal_idxs]] = 1
                class_loss = sigmoid_focal_loss_jit(
                    logits_pred,
                    focal_loss_class_target,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="none",
                )
                if self.class_denorm_type == "all":
                    class_loss = (class_loss * empty_weight).sum() / transformer_num_pos_avg
                elif self.class_denorm_type == "mixed":
                    class_loss = (class_loss * empty_weight).sum() / class_loss_denorm
                elif self.class_denorm_type == "mixed_2x":
                    class_loss = (class_loss * empty_weight).sum() / (2 * class_loss_denorm)
                elif self.class_denorm_type == "set":
                    class_loss = (class_loss * empty_weight).sum() / transformer_num_set_pos_avg
                else:
                    raise NotImplementedError

                correctness = logits_pred.argmax(dim=-1) == gt_classes
                acc_foreground = 0.0
                if transformer_pos_inds.numel() > 0:
                    acc_foreground = correctness[fg_inds]
                    acc_foreground = acc_foreground.sum().item() / (0.0 + acc_foreground.shape[0])
                extended_logits = torch.cat([logits_pred, torch.full_like(logits_pred, 0.1)[..., -1:]], dim=-1)
                correctness = extended_logits.argmax(dim=-1) == gt_classes
                acc_background = 0.0
                if transformer_pos_inds.numel() < class_target.shape[0]:
                    acc_background = correctness[gt_classes == num_classes]
                    acc_background = acc_background.sum().item() / (0.0 + acc_background.shape[0])

                extended_logits = torch.cat([logits_pred, torch.full_like(logits_pred, 0.5)[..., -1:]], dim=-1)
                correctness = extended_logits.argmax(dim=-1) == class_target
                acc_set_prediction_fg = 0.0
                if matched_proposal_idxs is not None:
                    acc_set_prediction_fg = correctness[(class_target != num_classes)]
                    acc_set_prediction_fg = acc_set_prediction_fg.sum().item() / (0.0 + acc_set_prediction_fg.shape[0])
                acc_set_prediction_bg = 0.0
                if (fg_inds & (class_target == num_classes)).sum().item() > 0:
                    acc_set_prediction_bg = correctness[fg_inds & (class_target == num_classes)]
                    acc_set_prediction_bg = acc_set_prediction_bg.sum().item() / (0.0 + acc_set_prediction_bg.shape[0])

                num_gpus = get_world_size()
                total_acc_foreground = reduce_sum(
                    logits_pred.new_tensor([acc_foreground])).item()
                acc_foreground_avg = total_acc_foreground / num_gpus
                total_acc_background = reduce_sum(
                    logits_pred.new_tensor([acc_background])).item()
                acc_background_avg = total_acc_background / num_gpus
                total_acc_set_prediction_fg = reduce_sum(
                    logits_pred.new_tensor([acc_set_prediction_fg])).item()
                acc_set_prediction_fg_avg = total_acc_set_prediction_fg / num_gpus
                total_acc_set_prediction_bg = reduce_sum(
                    logits_pred.new_tensor([acc_set_prediction_bg])).item()
                acc_set_prediction_bg_avg = total_acc_set_prediction_bg / num_gpus
            else:
                class_target = torch.zeros_like(logits_pred)
                class_target[transformer_pos_inds, transformer_labels[transformer_pos_inds]] = 1
                class_loss = sigmoid_focal_loss_jit(
                    logits_pred,
                    class_target,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                ) / transformer_num_pos_avg

            storage = get_event_storage()
            storage.put_scalar("fcos/loss_denorm", loss_denorm)
            storage.put_scalar("fcos/transformer_num_pos_avg", transformer_num_pos_avg)
            storage.put_scalar("fcos/num_pos_avg", num_pos_avg)
            if transformer_num_set_pos_avg is not None:
                storage.put_scalar("fcos/transformer_num_set_pos_avg", transformer_num_set_pos_avg)
            if class_loss_denorm is not None:
                storage.put_scalar("fcos/class_loss_denorm", class_loss_denorm)
            if self.use_obj_loss:
                storage.put_scalar("fcos/acc_foreground_avg", acc_foreground_avg)
                storage.put_scalar("fcos/acc_background_avg", acc_background_avg)
                storage.put_scalar("fcos/acc_set_prediction_fg_avg", acc_set_prediction_fg_avg)
                storage.put_scalar("fcos/acc_set_prediction_bg_avg", acc_set_prediction_bg_avg)

            if transformer_pos_inds.numel() > 0:
                reg_loss = self.loc_loss_func(
                    reg_pred,
                    instances.reg_targets,
                    ctrness_targets
                ) / loss_denorm

                ctrness_loss = F.binary_cross_entropy_with_logits(
                    ctrness_pred,
                    ctrness_targets,
                    reduction="sum"
                ) / transformer_num_pos_avg
            else:
                reg_loss = reg_pred.sum() * 0
                ctrness_loss = ctrness_pred.sum() * 0

            losses = {
                "loss_fcos_local_cls": local_class_loss,
                "loss_fcos_cls": class_loss,
                "loss_fcos_loc": reg_loss,
                "loss_fcos_ctr": ctrness_loss
            }
            extras = {
                "instances": instances,
                "loss_denorm": loss_denorm
            }
        return extras, losses

    def predict_proposals(
        self, logits_pred, reg_pred, ctrness_pred,
        locations, image_sizes, top_feats=None, attention_maps=None, visualize=False, fpn_levels=None,
    ):
        if self.training:
            self.pre_nms_thresh = self.pre_nms_thresh_train
            self.pre_nms_topk = self.pre_nms_topk_train
            self.post_nms_topk = self.post_nms_topk_train
        else:
            self.pre_nms_thresh = self.pre_nms_thresh_test
            self.pre_nms_topk = self.pre_nms_topk_test
            self.post_nms_topk = self.post_nms_topk_test

        sampled_boxes = []

        bundle = {
            "l": locations, "o": logits_pred,
            "r": reg_pred, "c": ctrness_pred,
            "s": self.strides, "a": attention_maps,
        }

        if len(top_feats) > 0:
            bundle["t"] = top_feats

        for i, per_bundle in enumerate(zip(*bundle.values())):
            # get per-level bundle
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            l = per_bundle["l"]
            o = per_bundle["o"]

            b = None
            if visualize:
                b = torch.ones_like(per_bundle["r"])
            r = per_bundle["r"] * per_bundle["s"]
            c = per_bundle["c"]
            t = per_bundle["t"] if "t" in bundle else None

            a = None
            if per_bundle["a"] is not None:
                a = per_bundle["a"].permute(0, 2, 1, 3)

            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, r, c, image_sizes, t, attention=a, original_box=b,
                )
            )

            for per_im_sampled_boxes in sampled_boxes[-1]:
                per_im_sampled_boxes.fpn_levels = l.new_ones(
                    len(per_im_sampled_boxes), dtype=torch.long
                ) * i

                if fpn_levels is not None:
                    per_im_sampled_boxes.real_fpn_levels = fpn_levels.view(-1).repeat(len(per_im_sampled_boxes), 1)

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    def forward_for_single_feature_map(
        self, locations, logits_pred, reg_pred,
        ctrness_pred, image_sizes, top_feat=None, attention=None, original_box=None
    ):
        N, P, C = logits_pred.shape

        # put in the same format as locations
        # logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)
        # logits_pred = logits_pred.reshape(N, -1, C).sigmoid()
        # box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        # box_regression = box_regression.reshape(N, -1, 4)
        # ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        # ctrness_pred = ctrness_pred.reshape(N, -1).sigmoid()

        logits_pred = logits_pred.sigmoid()
        box_regression = reg_pred
        ctrness_pred = ctrness_pred.view(N, -1).sigmoid()

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr and not self.predict_without_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]
        candidate_inds = logits_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)

        if not self.thresh_with_ctr and not self.predict_without_ctr:
            logits_pred = logits_pred * ctrness_pred[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = logits_pred[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]

            per_original_box = None
            per_original_locations = None
            if original_box is not None:
                per_original_locations = locations[i]
                per_original_box = original_box[i]

            per_attention = None
            if attention is not None:
                per_attention = attention[i]
                per_attention = per_attention[per_box_loc]

            per_locations = locations[i][per_box_loc]
            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                if per_attention is not None:
                    per_attention = per_attention[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations

            if per_original_box is not None:
                orig_detections = torch.stack([
                    per_original_locations[:, 0] - per_original_box[:, 0],
                    per_original_locations[:, 1] - per_original_box[:, 1],
                    per_original_locations[:, 0] + per_original_box[:, 2],
                    per_original_locations[:, 1] + per_original_box[:, 3],
                ], dim=1)
                orig_detections = orig_detections.repeat(detections.shape[0], 1, 1)
                boxlist.orig_boxes = orig_detections

            if per_attention is not None:
                boxlist.attentions = per_attention
            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_topk > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
