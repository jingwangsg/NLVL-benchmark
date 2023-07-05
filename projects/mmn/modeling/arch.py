import torch
from torch import nn
from torch.functional import F


class MMN(nn.Module):
    def __init__(self, featpool, feat2d, text_encoder, proposal_conv, criterion, joint_space_size, encoder_name, only_iou_loss_epoch=0):
        super(MMN, self).__init__()

        self.featpool = featpool
        self.feat2d = feat2d
        self.text_encoder = text_encoder
        self.proposal_conv = proposal_conv
        self.criterion = criterion

        self.join_space_size = joint_space_size
        self.encoder_name = encoder_name
        self.only_iou_loss_epoch = only_iou_loss_epoch

    def forward(self, vid_feat, txt_inds, txt_mask, se_times, gt_map=None, **kwargs):
        """
        Args:
            vid_feat: (Bv, Lv, Dv)
            txt_inds: (Nt, Lt)
            txt_mask: (Nt, Lt)
            se_times: (Bv, Lc, Lc)
            gt_map: (Bv, Lc, Lc)
        
        Returns:
        """
        Bv = vid_feat.shape[0]
        device = vid_feat.device

        txt_feat = self.text_encoder(txt_inds)
        txt_mask = txt_mask.unsqueeze(-1).float()

        vis_h = self.featpool(vid_feat.transpose(1, 2))
        map_h, map_mask = self.prop_module(vis_h)
