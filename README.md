# Rethinking Transformer-based Set Prediction for Object Detection

Here are the code for [the ICCV paper](https://arxiv.org/abs/2011.10881). The code is adapted from [Detectron2](https://github.com/facebookresearch/detectron2) and [AdelaiDet](https://github.com/aim-uofa/AdelaiDet).

The model is trained on 4 V100 GPUs.

## Prerequisites

Modify the environment name and environment prefix in `environment.yml` and run

```bash
conda env create -f environment.yml
```

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git reset --hard b88c6c06563e4db1139aafbd6d8d97d1fa7a57e4
pip install -e .
```

## Rreproducing Results

For TSP-FCOS,

```bash
bash tsp_fcos.sh
```

For TSP-RCNN,

```bash
bash tsp_rcnn.sh
```

## Citation
```
@InProceedings{Sun_2021_ICCV,
    author    = {Sun, Zhiqing and Cao, Shengcao and Yang, Yiming and Kitani, Kris M.},
    title     = {Rethinking Transformer-Based Set Prediction for Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {3611-3620}
}
```
