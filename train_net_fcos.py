#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script is a simplified version of the training script in detectron2/tools.
"""

import os
import random
import numpy as np
import torch
import time
import math
import logging


import pickle
from fvcore.common.file_io import PathManager

from collections import OrderedDict
from itertools import count
from typing import Any, Dict, List, Set
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
from rcnn import add_rcnn_config, DetrDatasetMapper
from fcos import add_fcos_config

from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from torch.nn.parallel import DistributedDataParallel
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling import GeneralizedRCNNWithTTA, DatasetMapperTTA
from rcnn.my_fast_rcnn_output import fast_rcnn_inference_single_image

from contextlib import ExitStack, contextmanager

from detectron2.data import detection_utils as utils
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class HybridOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        betas=betas, eps=eps, weight_decay=weight_decay)
        super(HybridOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HybridOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("optimizer", "SGD")

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if group["optimizer"] == "SGD":
                    weight_decay = group['weight_decay']
                    momentum = group['momentum']
                    dampening = group['dampening']

                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        d_p = buf
                    p.add_(d_p, alpha=-group['lr'])

                elif group["optimizer"] == "ADAMW":
                    # Perform stepweight decay
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                    # Perform optimization step
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    step_size = group['lr'] / bias_correction1

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    raise NotImplementedError

        return loss


class AdetCheckpointer(DetectionCheckpointer):
    """
    Same as :class:`DetectronCheckpointer`, but is able to convert models
    in AdelaiDet, such as LPF backbone.
    """
    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                if "weight_order" in data:
                    del data["weight_order"]
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}

        basename = os.path.basename(filename).lower()
        if "lpf" in basename or "dla" in basename:
            loaded["matching_heuristics"] = True
        return loaded


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        self.clip_norm_val = 0.0
        if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
                self.clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE

        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        super(DefaultTrainer, self).__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = AdetCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def run_step(self):
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        if self.clip_norm_val > 0.0:
            clipped_params = []
            for name, module in self.model.named_modules():
                for key, value in module.named_parameters(recurse=False):
                    if "transformer" in name:
                        clipped_params.append(value)
            torch.nn.utils.clip_grad_norm_(clipped_params, self.clip_norm_val)
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        for name, _ in model.named_modules():
            print(name)

        for name, module in model.named_modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                optimizer_name = "SGD"
                if isinstance(module, norm_module_types):
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
                elif key == "bias":
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

                if "bottom_up" in name:
                    lr = lr * cfg.SOLVER.BOTTOM_UP_MULTIPLIER
                elif "transformer" in name:
                    lr = lr * cfg.SOLVER.TRANSFORMER_MULTIPLIER
                    optimizer_name = "ADAMW"

                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "optimizer": optimizer_name}]

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        elif optimizer_type == "ADAMW":
            optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR)
        elif optimizer_type == "HYBRID":
            optimizer = HybridOptimizer(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it, load all checkpointables
        (eg. optimizer and scheduler) and update iteration counter.
        Otherwise, load the model specified by the config (skip all checkpointables) and start from
        the first iteration.
        Args:
            resume (bool): whether to do resume or not
        """
        path = self.cfg.MODEL.WEIGHTS
        if resume and self.checkpointer.has_checkpoint():
            path = self.checkpointer.get_checkpoint_file()
            checkpointables = [key for key in self.checkpointer.checkpointables.keys() if key != "scheduler"]
            checkpoint = self.checkpointer.load(path, checkpointables=checkpointables)
            for i in range(checkpoint.get("iteration", -1) + 1):
                self.checkpointer.checkpointables["scheduler"].step()
        else:
            checkpoint = self.checkpointer.load(path, checkpointables=[])

        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.CROP.ENABLED:
            mapper = DetrDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def visualize(cls, cfg, model, evaluators=None, dirname=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
            dirname: string
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        scale = 1.0
        if dirname is None:
            raise NotImplementedError

        def export_output(vis, fname):
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

        with ExitStack() as stack:
            if isinstance(model, torch.nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            for idx, batch in enumerate(data_loader):
                if idx < 50:
                    continue

                outputs = model(batch)

                for per_image, output in zip(batch, outputs):
                    proposals = output['instances'].orig_boxes
                    fpn_levels = output['instances'].real_fpn_levels
                    predictions = output['instances'].pred_boxes.tensor
                    all_attentions = output['instances'].attentions

                    strides = cfg.MODEL.FCOS.FPN_STRIDES

                    for head in range(48):
                        attentions = all_attentions[head]

                        print("predictions:", str(predictions.shape))
                        print("proposal:", str(proposals.shape))
                        print("attention:", str(attentions.shape))
                        print("fpn_levels:", str(fpn_levels.shape))

                        img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                        img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

                        visualizer = Visualizer(img, metadata=metadata, scale=scale)

                        for i, edge_color in zip([0, 8, 5, 2, 1, 4], ["r", "g", "b", "c", "m", "y"]):
                            box_coord = (predictions[i][0], predictions[i][1], predictions[i][2], predictions[i][3])
                            visualizer.output.scale = 1.0
                            visualizer.draw_box(box_coord, edge_color=edge_color)

                            # orig_box_coord = (
                            # proposals[i][0], proposals[i][1], proposals[i][2],
                            # proposals[i][3])
                            # visualizer.output.scale = 1.0
                            # visualizer.draw_box(orig_box_coord, edge_color=edge_color, line_style="-.")

                            visualizer.output.scale = 0.5
                            attention_boxes = torch.argsort(attentions[i], descending=True)[:5]
                            for j in range(5):
                                box_id = attention_boxes[j]
                                x1 = proposals[i][box_id][0]
                                y1 = proposals[i][box_id][1]
                                x2 = proposals[i][box_id][2]
                                y2 = proposals[i][box_id][3]

                                w = (x2 - x1) * strides[fpn_levels[i][box_id]]
                                h = (y2 - y1) * strides[fpn_levels[i][box_id]]
                                mean_x = (x1 + x2) / 2
                                mean_y = (y1 + y2) / 2

                                x1 = mean_x - (w / 2)
                                x2 = mean_x + (w / 2)
                                y1 = mean_y - (h / 2)
                                y2 = mean_y + (h / 2)

                                relevant_box = (x1, y1, x2, y2)
                                visualizer.draw_box(relevant_box, edge_color=edge_color, line_style="--")

                        # for i, edge_color in zip([12, 1, 6, 4, 100, 14], ["r", "g", "b", "c", "m", "y"]):
                        #     box_coord = (proposals[i][0], proposals[i][1], proposals[i][2], proposals[i][3])
                        #     visualizer.output.scale = 1.0
                        #     visualizer.draw_box(box_coord, edge_color=edge_color)
                        #
                        #     visualizer.output.scale = 0.5
                        #     attention_boxes = torch.argsort(attentions[i], descending=True)[:5]
                        #     for j in range(5):
                        #         box_id = attention_boxes[j]
                        #         relevant_box = (proposals[box_id][0], proposals[box_id][1], proposals[box_id][2], proposals[box_id][3])
                        #         visualizer.draw_box(relevant_box, edge_color=edge_color, line_style="--")

                        # for i in range(10):
                        #     box_coord = (predictions[i][0], predictions[i][1], predictions[i][2], predictions[i][3])
                        #     # orig_box_coord = (proposals[i][0], proposals[i][1], proposals[i][2], proposals[i][3])
                        #     visualizer.draw_box(box_coord)
                        #     visualizer.draw_text("%d" % i,
                        #                          [_ + 5 for _ in box_coord[:2]])
                        #     # visualizer.draw_box(orig_box_coord, edge_color="r")
                        #     # visualizer.draw_text("%d" % i,
                        #     #                      [_ + 5 for _ in orig_box_coord[:2]])

                        vis = visualizer.output

                        export_output(vis, str(per_image["image_id"]) + ("_head%d.jpg" % head))
                if idx == 50:
                    break

        return None


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


class MyGeneralizedRCNNWithTTA(GeneralizedRCNNWithTTA):
    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
        Args:
            cfg (CfgNode):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__(cfg, model, tta_mapper, batch_size)
        if isinstance(model, DistributedDataParallel):
            model = model.module
        assert isinstance(
            model, GeneralizedRCNN
        ), "TTA is only supported on GeneralizedRCNN. Got a model of type {}".format(type(model))
        self.cfg = cfg.clone()
        assert not self.cfg.MODEL.KEYPOINT_ON, "TTA for keypoint is not supported yet"
        assert (
            not self.cfg.MODEL.LOAD_PROPOSALS
        ), "TTA for pre-computed proposals is not supported yet"

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def _merge_detections(self, all_boxes, all_scores, all_classes, shape_hw):
        # select from the union of all results
        num_boxes = len(all_boxes)
        num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # +1 because fast_rcnn_inference expects background scores as well
        all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
        for idx, cls, score in zip(count(), all_classes, all_scores):
            all_scores_2d[idx, cls] = score

        merged_instances, _ = fast_rcnn_inference_single_image(
            all_boxes,
            all_scores_2d,
            shape_hw,
            self.cfg.MODEL.ROI_HEADS.TTA_SCORE_THRESH_TEST,
            self.cfg.MODEL.ROI_HEADS.TTA_NMS_THRESH_TEST,
            self.cfg.TEST.DETECTIONS_PER_IMAGE,
            self.cfg.MODEL.ROI_HEADS.TTA_SOFT_NMS_ENABLED,
            self.cfg.MODEL.ROI_HEADS.TTA_SOFT_NMS_METHOD,
            self.cfg.MODEL.ROI_HEADS.TTA_SOFT_NMS_SIGMA,
            self.cfg.MODEL.ROI_HEADS.TTA_SOFT_NMS_PRUNE,
        )

        return merged_instances


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_rcnn_config(cfg)
    add_fcos_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    os.environ['PYTHONHASHSEED'] = str(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    print("Random Seed:", cfg.SEED)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if args.my_visualize:
            res = Trainer.visualize(cfg, model, dirname=args.visualize_output)
        else:
            res = Trainer.test(cfg, model)
        return res

    # if cfg.MODEL.WEIGHTS.startswith("detectron2://ImageNetPretrained"):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--my-visualize", action="store_true",
                        help="perform visualization only")
    parser.add_argument("--visualize-output", default=None, type=str,
                        help="perform visualization only")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
