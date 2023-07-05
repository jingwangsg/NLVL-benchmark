import torch
from einops import einsum

def cosine_similarity(x, y, eps=1e-8):
    """
    Args:
        x: (L1, D)
        y: (L2, D)
    Return:
        sim: (L1, L2)

    """
    x_norm = x.norm(dim=-1, keepdim=True, p=2)
    y_norm = y.norm(dim=-1, keepdim=True, p=2).transpose(0, 1)
    sim = einsum(x, y, "l d, k d -> l k") / (x_norm * y_norm + eps)
    return sim


def inverse_sigmoid(x, eps=1e-3):
    """
    The inverse function for sigmoid activation function.
    Note: It might face numberical issues with fp16 small eps.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def cw2se(cw):
    s = cw[:, 1] - cw[:, 0] * 0.5
    t = cw[:, 1] + cw[:, 0] * 0.5
    s[s < 0] = 0
    t[t > 1] = 1
    return torch.stack([s, t], dim=1)


def calc_iou(pred_bds, gt, type="iou", eps=1e-6):
    """
    Args:
        calculate iou or giou in batch, make sure the range between [0, 1) to make loss function happy
        pred_bds: [*, 2] torch.float
        gt: [*, 2] torch.float
    Returns:
        iou: [*, ] torch.float
    """

    min_ed = torch.minimum(pred_bds[..., 1], gt[..., 1])
    max_ed = torch.maximum(pred_bds[..., 1], gt[..., 1])
    min_st = torch.minimum(pred_bds[..., 0], gt[..., 0])
    max_st = torch.maximum(pred_bds[..., 0], gt[..., 0])

    I = torch.maximum(min_ed - max_st, torch.zeros_like(min_ed, dtype=torch.float, device=pred_bds.device))
    area_pred = pred_bds[..., 1] - pred_bds[..., 0]
    area_gt = gt[..., 1] - gt[..., 0]
    U = area_pred + area_gt - I
    Ac = max_ed - min_st

    iou = I / (U + eps)

    if type == "iou":
        return iou
    elif type == "giou":
        return 0.5 * (iou + U / Ac)
    else:
        raise NotImplementedError()