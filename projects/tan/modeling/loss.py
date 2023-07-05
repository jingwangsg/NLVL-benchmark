import torch
import torch.nn.functional as F


def build_criterion(min_iou, max_iou, bias):

    def criterion(scores, masks, targets, return_prob=False):
        loss_value, joint_prob = bce_rescale_loss(scores, masks, targets, min_iou, max_iou, bias)
        return loss_value if not return_prob else (loss_value, joint_prob)

    return criterion


def bce_rescale_loss(scores, masks, targets, min_iou, max_iou, bias):
    joint_prob = scores
    target_prob = (targets - min_iou) * (1 - bias) / (max_iou - min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0

    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
    loss_value = torch.sum(loss) / torch.sum(masks)
    return loss_value, joint_prob
