import torch
import torch.nn.functional as F

def focal_loss_weight(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2,
):
    p = inputs.softmax(dim=-1)
    num_classes = inputs.shape[-1]
    gt_labels_target = F.one_hot(targets, num_classes=num_classes)

    p_t = (p * gt_labels_target).sum(-1)
    weight = gamma * ((1 - p_t) ** gamma)
    weight = weight.clone().detach()

    fg_inds = (targets >= 0) & (targets < num_classes - 1)
    weight[fg_inds] = 1.0

    return weight
