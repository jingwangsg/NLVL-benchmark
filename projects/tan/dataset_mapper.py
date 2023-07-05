from data.dataset_mapper import DatasetMapperByPair
import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from utils.math import calc_iou


def prepare_se_times(num_clips):
    s_times = torch.arange(0, num_clips, dtype=torch.float) / num_clips
    e_times = torch.arange(1, num_clips + 1, dtype=torch.float) / num_clips
    se_times = torch.stack(
        [repeat(s_times, "nc -> nc nc0", nc0=num_clips),
         repeat(e_times, "nc -> nc0 nc", nc0=num_clips)], dim=-1)  # [B, num_clips, num_clips, 2]
    return se_times


def prepare_mapgt(gt, se_times):
    """ prepare ground truth map for training
    Args:
        gt: [B, 2] normalized ground truth
        se_times: [num_clips, num_clips, 2] start and end time for each moment
    Returns:
        ious: [B, num_clips, num_clips] IoU between each clip and ground truth,
            where the second dimension is start time and the third dimension is end time
    """
    B = gt.shape[0]
    Nc = se_times.shape[1]

    se_times_repeat = se_times
    gt_repeat = repeat(gt, "b i -> b nc nc0 i", nc=Nc, nc0=Nc)
    ious = calc_iou(se_times_repeat, gt_repeat)
    return ious.unsqueeze(1)


class DatasetMapperTAN(DatasetMapperByPair):

    def __init__(self, video_pipe, text_pipe, num_clips):
        super().__init__(video_pipe, text_pipe)
        self.num_clips = num_clips

    def __call__(self, batch):
        result = super().__call__(batch)
        B = result["gt"].shape[0]
        result["se_times"] = prepare_se_times(self.num_clips).unsqueeze(0).repeat(B, 1, 1, 1)
        result["gt_map"] = prepare_mapgt(result["gt"], result["se_times"])
        return result