import torch
from torch import nn
import torch.nn.functional as F


class BaseFusion(nn.Module):

    def __init__(self, hidden_size, txt_input_size, txt_hidden_size, bidirecitonal=True, num_layers=1):
        """ fuse textual and visual features
        Args:
            hidden_size: output feature size
            txt_input_size: input feature size of textual features
            txt_hidden_size: hidden feature size in textual encoder
            bidirecitonal: whether use bidirectional lstm
            num_layers: number of layers in textual encoder
        Returns:
            fused_h: fused features
        """
        super(BaseFusion, self).__init__()
        self.textual_encoder = nn.LSTM(txt_input_size,
                                       txt_hidden_size // 2 if bidirecitonal else txt_hidden_size,
                                       num_layers=num_layers,
                                       bidirectional=bidirecitonal,
                                       batch_first=True)
        self.tex_linear = nn.Linear(txt_hidden_size, hidden_size)
        self.vis_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1)

    def forward(self, textual_input, textual_mask, map_h, map_mask):
        self.textual_encoder.flatten_parameters()
        txt_h = self.textual_encoder(textual_input)[0] * textual_mask
        txt_h = torch.stack([txt_h[i][torch.sum(mask).long() - 1] for i, mask in enumerate(textual_mask)])
        txt_h = self.tex_linear(txt_h)[:, :, None, None]
        map_h = self.vis_conv(map_h)
        fused_h = F.normalize(txt_h * map_h) * map_mask
        return fused_h
