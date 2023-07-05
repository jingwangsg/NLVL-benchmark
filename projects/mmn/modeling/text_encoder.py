import torch
from torch import nn
from transformers import DistilBertModel


class TextEncoderDistill(nn.Module):

    def __init__(self, d_model) -> None:
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc_out1 = nn.Linear(768, d_model)
        self.fc_out2 = nn.Linear(768, d_model)
        self.layernorm = nn.LayerNorm(768)
        self.aggregation = "avg"  # cls, avg

    def forward(self, txt_inds, txt_mask):
        """
        Args:
            txt_inds: (Nt, Lt)
            txt_mask: (Nt, Lt)
        """
        bert_hidden = self.bert(txt_inds, attention_mask=txt_mask)[0]  # (Nt, Lt, D)

        if self.aggregation == "cls":
            sent_feat = bert_hidden[:, 0, :]  # [Nt, D], use [CLS] (first token) as the whole sentence feature
        elif self.aggregation == "avg":
            sent_feat = bert_hidden.sum(1) / txt_mask.sum(1)  # [Nt, D]

        sent_feat = self.layernorm(sent_feat)
        out_feat = self.fc_out1(sent_feat)
        out_feat_iou = self.fc_out2(sent_feat)

        return out_feat, out_feat_iou