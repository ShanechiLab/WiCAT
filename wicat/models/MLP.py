from typing import List

import torch.nn as nn

from wicat.utility.torch_utils import get_activation_function


class MLP(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_out: int,
        layer_list: List = None,
        dropout: float = 0.1,
        use_final_dropout: bool = False,
        activation: str = "linear",
        use_identity: bool = False,
        **kwargs
    ):
        super(MLP, self).__init__()

        self.d_input = d_input
        self.d_out = d_out
        self.d_hidden = d_out
        self.layer_list = layer_list
        self.dropout = dropout
        self.use_final_dropout = use_final_dropout
        self.use_identity = use_identity
        self.activation_fn = get_activation_function(activation)

        if self.use_identity:
            self.layers = nn.ModuleList([nn.Identity()])
            self.final_layer = nn.Identity()
            return

        current_dim = self.d_input
        self.layers = nn.ModuleList()
        if self.layer_list is not None:
            for _, dim in enumerate(self.layer_list):
                self.layers.append(nn.Linear(current_dim, dim))
                current_dim = dim
        else:
            self.layers.append(nn.Identity())

        self.final_layer = nn.Linear(current_dim, self.d_out)

    def forward(self, x, **kwargs):
        if self.use_identity:
            return x
        x = nn.Dropout(self.dropout)(x)
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
            x = nn.Dropout(self.dropout)(x)
        x = self.final_layer(x)
        if self.use_final_dropout:
            x = nn.Dropout(self.dropout)(x)
        return x
