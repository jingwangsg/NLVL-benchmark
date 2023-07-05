import torch
import math
import torch.nn as nn
from typing import List
from einops import repeat


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
) -> torch.Tensor:
    """generate sine position embedding from a position tensor
    formula from Vasawa et al. https://arxiv.org/pdf/1706.03762.pdf

    PE(pos, 2i) = sin(pos / temp^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / temp^(2i / d_model))

    Args:
        pos_tensor (torch.Tensor): arbitrary shape tensor
        num_pos_feats (int): projected shape for each float in the tensor. Default: 128
        temperature (int): The temperature used for scaling
            the position embedding. Default: 10000.

    Returns:
        torch.Tensor: Returned position embedding  # noqa 
        pos_tensor (*, ), return tensor with shape (*, num_pos_feats)
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature**(2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    orig_shape = pos_tensor.shape
    pos_tensor_flat = pos_tensor.reshape(-1)

    pos_sin_flat = pos_tensor_flat[:, None] * scale / dim_t[None, :]
    pos_res = torch.stack((pos_sin_flat[:, 0::2].sin(), pos_sin_flat[:, 1::2].cos()), dim=1).transpose(1, 2).flatten(1)

    pos_res = pos_res.reshape((*orig_shape, -1))

    return pos_res


class PositionalEncodingLearn(nn.Module):

    def __init__(self, hidden_size: int = 256, max_length: List[int] = [128]) -> None:
        """ a general learnable positional encoding for aribitrary dimension
        Args:
            hidden_size (int, optional): hidden size for the positional encoding. Defaults to 256.
            max_length (List[int], optional): max length for each dimension. Defaults to [128].
                number of elements in the list is the number of dimensions
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embeds = nn.ModuleList([nn.Embedding(max_len, hidden_size) for max_len in max_length])

    def forward(self, x: torch.Tensor, is_batch: bool = True, flatten: bool = True):
        """forward function for the positional encoding
        for example, if x.shape = (3,4,5) (is_batch=False) then the positional encoding will be (3,4,5,3,hidden_size)
        Args:
            x (torch.Tensor): input tensor with shape (bs, *) or (*, )
            is_batch (bool, optional): whether the input tensor is batched. Defaults to True.
        Return:
            torch.Tensor: positional encoding with shape (bs, *, hidden_size) or (*, hidden_size)
        """

        shp = list(x.shape)[1:] if is_batch else list(x.shape)
        dims = len(shp)
        tensor_list = []
        for i, l in enumerate(shp):
            emb = self.embeds[i].weight[:l]
            from_axes = [f"ax{i}", "D"]
            to_axes = [f"ax{k}" for k in range(dims)]
            to_shape = {to_axes[i]: shp[i] for i in range(dims)}

            to_axes = to_axes + ["D"]

            if is_batch:
                # from_axes = ["b"] + from_axes
                to_axes = ["b"] + to_axes
                to_shape["b"] = x.shape[0]

            from_axes = " ".join(from_axes)
            to_axes = " ".join(to_axes)

            emb = repeat(emb, from_axes + "->" + to_axes, **to_shape)
            tensor_list.append(emb)

        emb = torch.stack(tensor_list, dim=-2)

        if flatten:
            emb = emb.flatten(-2)

        return emb


class PositionalEncodingSine(nn.Module):

    def __init__(self, d_model, temperature=10000):
        super().__init__()
        self.temperature = temperature
        self.d_model = d_model

    def forward(self, x, is_batch=True, flatten=False, scale=False):
        """ forward function for the sine positional encoding
        Args:
            x (torch.Tensor): input tensor with shape (bs, *) or (*, )
            is_batch (bool, optional): whether the input tensor is batched. Defaults to True.
            flatten (bool, optional): whether to flatten the output tensor. Defaults to False.
            scale (bool, optional): whether to scale the index to (0,1). Defaults to False.
        Return:
            emb (torch.Tensor): positional encoding with shape (bs, *, hidden_size) or (*, hidden_size)
        """
        shp = x.shape[1:-1] if is_batch else x.shape[:-1]  # do not include embedding dim
        dims = len(shp)
        index_list = []
        for i, l in enumerate(shp):
            index = torch.arange(l, device=x.device)
            if scale:
                index = index / l

            from_axes = [f"ax{i}"]
            to_axes = [f"ax{k}" for k in range(dims)]
            to_shape = {to_axes[i]: shp[i] for i in range(dims)}

            if is_batch:
                to_axes = ["b"] + to_axes
                to_shape["b"] = x.shape[0]

            from_axes = " ".join(from_axes)
            to_axes = " ".join(to_axes)

            index = repeat(index, from_axes + "->" + to_axes, **to_shape)
            index_list.append(index)

        index = torch.stack(index_list, dim=-1)

        emb = get_sine_pos_embed(index, num_pos_feats=self.d_model, temperature=self.temperature)
        if flatten:
            emb = emb.flatten(-2)

        return emb
