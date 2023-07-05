import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.positional_encoding import get_sine_pos_embed
from ..layers.attention import ConditionalCrossAttention, ConditionalSelfAttention
from ..layers.ffn import FFN
from kn_util.utils import clones
from utils.math import inverse_sigmoid
from einops import rearrange, repeat
from utils.math import cw2se
import os


class VideoDecoderLayerDabTrm(nn.Module):

    def __init__(self, d_model, nhead, ff_dim, dropout) -> None:
        super().__init__()
        self.self_attn = ConditionalSelfAttention(d_model,
                                                  nhead,
                                                  attn_drop=dropout,
                                                  proj_drop=dropout,
                                                  batch_first=True)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-12)
        self.cross_attn = ConditionalCrossAttention(d_model,
                                                    nhead,
                                                    attn_drop=dropout,
                                                    proj_drop=dropout,
                                                    batch_first=True)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FFN([d_model, ff_dim, d_model], use_identity=True)
        self.ln3 = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, key, query, value, query_sine_embed, query_pos, key_pos, is_first_layer=False):
        query = self.self_attn(query=query, key=query, value=query, query_pos=query_pos, key_pos=query_pos)
        query = self.ln1(query)
        query = self.cross_attn(query=query,
                                key=key,
                                value=value,
                                query_pos=query_pos,
                                key_pos=key_pos,
                                query_sine_embed=query_sine_embed,
                                is_first_layer=is_first_layer)
        query = self.ln2(query)
        query = self.ffn(query)
        query = self.ln3(query)

        return query


class VideoDecoderDabTrm(nn.Module):

    def __init__(self, d_model, nhead, ff_dim, num_queries, num_layers, dropout=0.1, share_refine=True) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        self.anchors = nn.Embedding(num_queries, 4)
        layer = VideoDecoderLayerDabTrm(d_model, nhead, ff_dim, dropout=dropout)
        self.layers = clones(layer, num_layers)
        self.ref_point_head = FFN([d_model, d_model, d_model])
        self.query_scale = FFN([d_model, d_model, d_model])  # query scale from Conditional DETR
        self.ref_anchor_head = FFN([d_model, d_model, 1])
        if not share_refine:
            self.bbox_refine_heads = clones(FFN([d_model, d_model, 2]), num_layers)
        else:
            self.bbox_refine_heads = nn.ModuleList([FFN([d_model, d_model, 2]) for _ in range(num_layers)])
        self.query_proj = FFN([d_model, d_model, d_model])

        self.init_weights()

    def init_weights(self):
        for refine_head in self.bbox_refine_heads:
            nn.init.constant_(refine_head.layers[-1].weight.data, 0)
            nn.init.constant_(refine_head.layers[-1].bias.data, 0)

    def forward(self, vid_feat, vid_pos, anchor_pos, **kwargs):
        """
        vid_feat: (B, Nv, D)
        anchor_pos: (B, Nq, 2)
        """
        B, Nv, D = vid_feat.shape
        Nq = anchor_pos.shape[1]

        query = torch.zeros(B, self.num_queries, D).to(vid_feat.device)

        reference_boxes = anchor_pos.sigmoid()  # (B, Nq, 2)
        intermediate_ref_boxes = []
        intermediate = []

        for idx, layer in enumerate(self.layers):
            query_sine_embed = get_sine_pos_embed(reference_boxes, self.d_model, temperature=10000)  # (B, Nq, 2, D)
            query_pos = self.ref_point_head(query_sine_embed[:, :, 0])  # (B, Nq, D)

            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)  # (B, Nq, D)

            # position_transform = 1

            query_sine_embed = query_sine_embed[:, :, 0] * position_transform

            # modulate size
            ref_cond = self.ref_anchor_head(query).sigmoid()
            query_sine_embed = query_sine_embed * ref_cond / reference_boxes[:, :, 1].unsqueeze(-1)

            query = layer(query=query,
                          key=vid_feat,
                          value=vid_feat,
                          key_pos=vid_pos,
                          query_pos=query_pos,
                          query_sine_embed=query_sine_embed,
                          is_first_layer=idx == 0)

            # refine boxes
            offset = self.bbox_refine_heads[idx](query)
            reference_boxes = inverse_sigmoid(reference_boxes)
            new_reference_boxes = (reference_boxes + offset).sigmoid()
            new_reference_boxes_flatten = rearrange(new_reference_boxes, "b nq i -> (b nq) i")
            new_reference_boxes_se = cw2se(new_reference_boxes_flatten).reshape(B, Nq, 2)
            intermediate_ref_boxes.append(new_reference_boxes_se)
            reference_boxes = new_reference_boxes.detach()
            intermediate.append(self.query_proj(query))

        return intermediate, intermediate_ref_boxes


class VideoDecoderTrm(nn.Module):

    def __init__(self, d_model, nhead, ff_dim, dropout, num_queries, num_layers) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model=d_model,
                                           nhead=nhead,
                                           dim_feedforward=ff_dim,
                                           dropout=dropout,
                                           batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.bbox_head = FFN([d_model, d_model, 2])
        self.query_emb = nn.Embedding(num_queries, d_model)

    def forward(self, vid_feat, vid_pos, **kwargs):
        B = vid_feat.shape[0]

        query = repeat(self.query_emb.weight, "nq d -> b nq d", b=B)
        query = self.decoder(memory=vid_feat, tgt=query, tgt_mask=None)

        ref_boxes = self.bbox_head(query).sigmoid()  # (B, Nq, 2)
        ref_boxes = cw2se(ref_boxes.reshape(-1, 2)).reshape(B, -1, 2)

        return [query], [ref_boxes]