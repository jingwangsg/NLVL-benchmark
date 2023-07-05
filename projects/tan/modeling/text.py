import torch
import torch.nn as nn
import torch.nn.functional as F
from kn_util.basic import global_get

class TextEncoderGlove(nn.Module):
    def __init__(self, global_glove_key="glove", freeze=False) -> None:
        super().__init__()
        vectors = global_get(global_glove_key).vectors
        self.embedding = nn.Embedding.from_pretrained(vectors, freeze=freeze)
    
    def forward(self, txt_inds):
        return self.embedding(txt_inds)