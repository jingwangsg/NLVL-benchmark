import torch
import torch.nn as nn
from .loss import bce_rescale_loss
from einops import rearrange, repeat
from .utils import nms
from kn_util.data import pad_sequence_general
from kn_util.debug import load_weight_compatible
from loguru import logger

def load_tan_compatible(model):
    mapping = {
        "video_encoder": "frame_layer",
        "prop_module": "prop_layer",
        "fusion_module": "fusion_layer",
        "map_conv": "map_layer",
        "predicter": "pred_layer",
    }
    state_dict = torch.load("/tmp/tan_state_dict.pt")
    return load_weight_compatible(model, state_dict, mapping=mapping)


class TAN(nn.Module):

    def __init__(self,
                 video_encoder,
                 text_encoder,
                 fusion_module,
                 map_conv,
                 prop_module,
                 criterion,
                 pred_input_size,
                 iou_threshold=0.5) -> None:
        super().__init__()
        self.video_encoder = video_encoder
        self.text_encoder = text_encoder
        self.fusion_module = fusion_module
        self.map_conv = map_conv
        self.prop_module = prop_module
        self.criterion = criterion
        self.predicter = nn.Conv2d(pred_input_size, 1, 1, 1)
        self.iou_threshold = iou_threshold

    def forward(self, txt_inds, txt_mask, vid_feat, se_times, gt_map=None, **kwargs):
        B = vid_feat.shape[0]
        device = vid_feat.device
        txt_feat = self.text_encoder(txt_inds)
        txt_mask = txt_mask.unsqueeze(-1).float()

        vis_h = self.video_encoder(vid_feat.transpose(1, 2))
        map_h, map_mask = self.prop_module(vis_h)

        fused_h = self.fusion_module(txt_feat, txt_mask, map_h, map_mask)
        fused_h = self.map_conv(fused_h, map_mask)
        prediction = self.predicter(fused_h)
        score = prediction.sigmoid() * map_mask

        loss = self.criterion(score, map_mask, gt_map)

        if self.training:
            loss_dict = dict()
            loss_dict["loss"] = loss
            return loss_dict
        else:
            num_clips = prediction.shape[2]
            score_flat = rearrange(score, "b c h w -> (b c h w)")
            se_times_flat = rearrange(se_times, "b h w i-> (b h w) i")
            batch_idxs_flat = repeat(torch.arange(B, device=device), "b -> (b k)", k=num_clips * num_clips)

            nms_pred_bds_list, nms_scores_list = nms(se_times_flat,
                                                     score_flat,
                                                     batch_idxs_flat,
                                                     iou_threshold=self.iou_threshold)
            # nms_pred_bds = pad_sequence_general(nms_pred_bds_list, fill_value=0.0, axis=0)
            nms_pred_bds = torch.stack([_[:10] for _ in nms_pred_bds_list], dim=0)
            return dict(timestamps_pred=nms_pred_bds, loss=loss)
