import torch
import torch.nn as nn
import torch.nn.functional as F
from kn_util.utils import clones

from .layers import Conv1D

class SeqPAN(nn.Module):

    def __init__(self, text_encoder, video_encoder, video_affine, dual_attention, cq_attention, cq_concatenate, predictor, d_model):
        super(SeqPAN, self).__init__()

        self.text_encoder = text_encoder

        # self.tfeat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4, max_pos_len=max_pos_len, droprate=droprate)

        self.video_affine = video_affine
        self.video_encoder = video_encoder
        self.dual_attention_blocks = clones(dual_attention, 2)
        self.dual_attention_block_1 = self.dual_attention_blocks[0]
        self.dual_attention_block_2 = self.dual_attention_blocks[1]

        self.cq_attentions = clones(cq_attention, 2)
        self.v2q_attn = self.cq_attentions[0]
        self.cq_cat = cq_concatenate
        self.match_conv1d = Conv1D(in_dim=d_model, out_dim=4)

        lable_emb = torch.empty(size=[d_model, 4], dtype=torch.float32)
        lable_emb = torch.nn.init.orthogonal_(lable_emb.data)
        self.label_embs = nn.Parameter(lable_emb, requires_grad=True)

        self.predictor = predictor

    def forward(self, word_ids, char_ids, vfeat_in, vmask, tmask):
        B = vmask.shape[0]
        tfeat = self.text_encoder(word_ids, char_ids)
        vfeat = self.video_affine(vfeat_in)

        vfeat = self.vfeat_encoder(vfeat)
        tfeat = self.vfeat_encoder(tfeat)

        vfeat_ = self.dual_attention_block_1(vfeat, tfeat, vmask, tmask)
        tfeat_ = self.dual_attention_block_1(tfeat, vfeat, tmask, vmask)
        vfeat, tfeat = vfeat_, tfeat_

        vfeat_ = self.dual_attention_block_2(vfeat, tfeat, vmask, tmask)
        tfeat_ = self.dual_attention_block_2(tfeat, vfeat, tmask, vmask)
        vfeat, tfeat = vfeat_, tfeat_

        t2v_feat = self.q2v_attn(vfeat, tfeat, vmask, tmask)
        v2t_feat = self.v2q_attn(tfeat, vfeat, tmask, vmask)
        fuse_feat = self.cq_cat(t2v_feat, v2t_feat, tmask)

        match_logits = self.match_conv1d(fuse_feat)
        match_score = F.gumbel_softmax(match_logits, tau=0.3)
        match_probs = torch.log(match_score)
        soft_label_embs = torch.matmul(match_score, torch.tile(self.label_embs, (B, 1, 1)).permute(0, 2, 1))
        fuse_feat = (fuse_feat + soft_label_embs) * vmask.unsqueeze(2)

        slogits, elogits = self.predictor(fuse_feat, vmask)
        return {
            "slogits": slogits,
            "elogits": elogits,
            "vmask": vmask,
            "match_score": match_score,
            "label_embs": self.label_embs,
        }
