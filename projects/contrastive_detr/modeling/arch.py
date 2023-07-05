import torch
import torch.nn as nn
from einops import repeat
from ..layers import FFN
from utils.math import cosine_similarity


class ContrastiveDetr(nn.Module):

    def __init__(
        self,
        txt_encoder: nn.Module,
        vid_encoder: nn.Module,
        vid_decoder: nn.Module,
        criterion: nn.Module,
        d_model: int,
        num_queries: int,
        num_pattern: int = 0,
    ):
        super().__init__()
        self.txt_encoder = txt_encoder
        self.vid_encoder = vid_encoder
        self.vid_decoder = vid_decoder
        self.criterion = criterion

        self.anchors = nn.Embedding(num_queries, 2)
        self.num_pattern = num_pattern

        # proj = FFN([d_model, d_model, d_model], use_identity=True)
        # self.projs = clones(proj, len(vid_decoder.layers))
        self.proj = FFN([d_model, d_model, d_model], use_identity=True)

    def forward(self, vid_feat, txt_inds, txt_mask, batch_split_size, gt=None, **kwargs):
        # for name, parameter in self.named_parameters():
        #     logger.info(f"{name}: {parameter.norm()}")

        B = vid_feat.shape[0]
        # logger.debug("vid_feat.original.norm: {:.4f}".format(vid_feat.norm()))
        txt_feat = self.txt_encoder(txt_inds=txt_inds, txt_mask=txt_mask)
        vid_feat, vid_pos = self.vid_encoder(vid_feat=vid_feat)
        # logger.debug("vid_feat.norm: {:.4f} txt_feat.norm: {:.4f}".format(vid_feat.norm(), txt_feat.norm()))

        anchor_pos = repeat(self.anchors.weight, "nq d -> b nq d", b=B)
        intermediate_query, intermediate_ref_boxes = self.vid_decoder(vid_feat=vid_feat,
                                                                      vid_pos=vid_pos,
                                                                      anchor_pos=anchor_pos)
        intermediate_query = [self.proj(q) for i, q in enumerate(intermediate_query)]

        if self.training:
            assert gt is not None, "gt is required for training"
            loss_dict = self.criterion(reference_boxes=intermediate_ref_boxes,
                                       query=intermediate_query,
                                       txt_feat=txt_feat,
                                       gt=gt,
                                       batch_split_size=batch_split_size)
            return loss_dict
        else:
            txt_feat_split = torch.split(txt_feat, batch_split_size)
            final_query = intermediate_query[-1]
            final_ref_boxes = intermediate_ref_boxes[-1]  # (B, Nq, 2)
            timestamps_pred_all = []
            for i in range(B):
                Nt_i = txt_feat_split[i].shape[0]
                sim = cosine_similarity(final_query[i], txt_feat_split[i])  # (Nq, Nt_i)
                topk_inds = sim.topk(k=10, dim=0).indices  # (10, Nt_i)
                timestamps_pred = final_ref_boxes[i, topk_inds]  # (10, Nt_i, 2)

                timestamps_pred_all.append(timestamps_pred)

            timestamps_pred = torch.cat(timestamps_pred_all, dim=1)  # (10, Nt, 2)
            timestamps_pred = timestamps_pred.transpose(0, 1)  # (Nt, 10, 2)

            ret = dict(timestamps_pred=timestamps_pred)
            if gt is not None:
                loss_dict = self.criterion(reference_boxes=intermediate_ref_boxes,
                                           query=intermediate_query,
                                           txt_feat=txt_feat,
                                           gt=gt,
                                           batch_split_size=batch_split_size)
                ret.update(loss_dict)

            return ret
