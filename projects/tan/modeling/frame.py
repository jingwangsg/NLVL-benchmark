import torch
import torch.nn as nn


class FrameAvgPool(nn.Module):
    """
    1. 1D Convolution through temporal dimension, same as linear projection
    2. Average Pooling with given stride and kernel_size
    """

    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super(FrameAvgPool, self).__init__()
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)
        self.stride = stride

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.avg_pool(vis_h)
        return vis_h


class FrameMaxPool(nn.Module):
    """
    1. 1D Convolution through temporal dimension, same as linear projection
    2. Max Pooling with given stride and kernel_size
    """

    def __init__(self, input_size, hidden_size, stride):
        super(FrameMaxPool, self).__init__()
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.max_pool = nn.MaxPool1d(stride)
        self.stride = stride

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.max_pool(vis_h)
        return vis_h
