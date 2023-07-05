from typing import Callable, List
import torch
import torch.nn as nn


def get_activition(activation: str, **kwargs) -> Callable:
    if activation == "relu":
        return nn.ReLU(**kwargs)
    elif activation == "gelu":
        return nn.GELU(**kwargs)
    elif activation == "tanh":
        return nn.Tanh(**kwargs)
    elif activation == "sigmoid":
        return nn.Sigmoid(**kwargs)
    else:
        raise NotImplementedError(f"Activation {activation} not implemented")


class FFN(nn.Module):

    def __init__(self,
                 dims: List,
                 activation: str = "relu",
                 dropout: float = 0.0,
                 use_ln: bool = False,
                 use_identity: bool = False,
                 disable_last_ln: bool = True,
                 disable_last_do: bool = True) -> None:
        super().__init__()
        self.num_layers = len(dims) - 1
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(self.num_layers)])
        self.activation = get_activition(activation.lower())

        self.dos = nn.ModuleList([nn.Dropout(dropout) for _ in range(self.num_layers)]) if dropout > 0 else None
        self.lns = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(self.num_layers)]) if use_ln else None
        self.disbale_last_ln = disable_last_ln
        self.disbale_last_do = disable_last_do

        self.use_identity = use_identity

    def forward(self, x, identity=None):
        if identity is None and self.use_identity:
            identity = x

        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            if self.lns is not None:
                x = self.lns[i](x)
            if self.dos is not None:
                x = self.dos[i](x)
            if self.activation is not None:
                x = self.activation(x)

        x = self.layers[-1](x)

        if self.use_identity:
            x = x + identity

        if self.lns is not None and not self.disbale_last_ln:
            x = self.lns[-1](x)

        if self.dos is not None and not self.disbale_last_do:
            x = self.dos[-1](x)

        return x
