import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from kn_util.basic import registry, global_get
from ..layers import FFN
from ..layers.positional_encoding import PositionalEncodingSine
from einops import rearrange, repeat


class TextEncoderPretrained(nn.Module):

    def __init__(self, d_model, pretrained="distilbert-base-uncased", agg="cls"):
        super().__init__()
        self.net = AutoModel.from_pretrained(pretrained)
        # self.w = nn.Linear(self.net.config.hidden_size, d_model)
        # self.ln = nn.LayerNorm(d_model, eps=1e-12)
        txt_input_dim = self.net.config.hidden_size
        self.mlp = FFN([txt_input_dim, d_model, d_model], dropout=0.1)
        self.agg = agg

    def forward(self, txt_inds, txt_mask):
        hidden_state = self.net(input_ids=txt_inds, attention_mask=(~txt_mask).float()).last_hidden_state
        if self.agg == "cls":
            hidden_state = hidden_state[:, 0]
        elif self.agg == "avg":
            hidden_state = hidden_state.mean(dim=1)
        return self.mlp(hidden_state)


class TextEncoderGloveTrm(nn.Module):

    def __init__(self, d_model, nhead, ff_dim, glove_key="glove", dropout=0.1, pooling="cls", num_layers=3) -> None:
        super().__init__()
        vectors = global_get(glove_key)["vectors"]
        self.embedding = nn.Embedding.from_pretrained(vectors)
        self.ln = nn.LayerNorm(d_model, eps=1e-12)
        self.pe = PositionalEncodingSine(d_model)
        self.do = nn.Dropout(dropout)

        self.pooling = pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(d_model))

        layer = nn.TransformerEncoderLayer(d_model=d_model,
                                           nhead=nhead,
                                           dim_feedforward=ff_dim,
                                           dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, txt_inds, txt_mask):
        Nt = txt_inds.shape[0]

        txt_feat = self.do(self.embedding(txt_inds))
        if self.pooling == "cls":
            txt_feat = torch.cat(repeat(self.cls_token, "d -> nt i d", nt=Nt, i=1), txt_feat, dim=1)

        txt_feat = self.ln(txt_feat + self.pe(txt_feat, flatten=True))
        txt_feat = self.encoder(txt_feat, src_key_padding_mask=~txt_mask)

        if self.pooling == "cls":
            txt_feat = txt_feat[:, 0]
        elif self.pooling == "mean":
            txt_feat = txt_feat.mean(dim=1)
        else:
            raise ValueError(f"pooling {self.pooling} not supported")

        return txt_feat


class VideoEncoderTrm(nn.Module):

    def __init__(self, d_model, vid_feat_dim, nhead, ff_dim, dropout, num_layers) -> None:
        super().__init__()
        self.w_vid = nn.Linear(vid_feat_dim, d_model)
        self.pe = PositionalEncodingSine(d_model)
        self.ln = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(d_model=d_model,
                                           nhead=nhead,
                                           dim_feedforward=ff_dim,
                                           dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, vid_feat, vid_mask=None):
        vid_mask = ~vid_mask if vid_mask is not None else None
        vid_pos = self.pe(vid_feat, flatten=True, scale=True)
        vid_feat = self.ln(self.w_vid(vid_feat) + vid_pos)
        vid_feat_ = self.encoder(vid_feat, src_key_padding_mask=vid_mask)

        return vid_feat_, vid_pos
