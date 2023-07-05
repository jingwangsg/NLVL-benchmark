import torch
from torchvision.ops import batched_nms

def nms(pred_bds, scores, batch_idxs, iou_threshold):
    """ nms in batch
    Args:
        pred_bds: (B, 2) [start, end]
        scores: (B, ) pred scores
        batch_idxs: (B, ) batch indices
        iou_threshold: float
    Returns:
        nms_pred_bds: list of (n, 2) [start, end]
        nms_scores: list of (n, ) pred scores
    """
    B, _2 = pred_bds.shape

    zero_pad = torch.zeros(pred_bds.shape[:1], dtype=torch.float32, device=pred_bds.device)
    one_pad = zero_pad + 1
    boxxes = torch.stack([pred_bds[:, 0], zero_pad, pred_bds[:, 1], one_pad], dim=-1)
    boxxes_flatten = boxxes
    scores_flatten = scores

    nms_indices = batched_nms(boxxes_flatten, scores_flatten, batch_idxs, iou_threshold)
    nms_pred_bds_flatten = boxxes_flatten[nms_indices][:, (0, 2)]
    nms_scores_flatten = scores_flatten[nms_indices]
    nms_idxs = batch_idxs[nms_indices]

    nms_pred_bds = []
    nms_scores = []
    for b in range(torch.max(batch_idxs).item() + 1):
        cur_batch_indices = (nms_idxs == b)
        nms_pred_bds.append(nms_pred_bds_flatten[cur_batch_indices])
        nms_scores.append(nms_scores_flatten[cur_batch_indices])

    return nms_pred_bds, nms_scores