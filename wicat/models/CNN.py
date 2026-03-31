from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from wicat.utility.torch_utils import get_activation_function


class CNN(nn.Module):
    def __init__(
        self,
        temporal_patch_size: int,
        spatial_patch_size: int,
        d_hidden: int,
        layer_list: List = None,
        dropout: float = 0.0,
        use_final_dropout: bool = False,
        activation: str = "leakyrelu",
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        **kwargs
    ):
        super(CNN, self).__init__()

        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.d_hidden = d_hidden
        self.layer_list = layer_list
        self.dropout = dropout
        self.use_final_dropout = use_final_dropout
        self.activation_fn = get_activation_function(activation)
        self.kernel_size = (1, kernel_size, kernel_size)
        self.stride = (1, stride, stride)
        self.padding = (0, padding, padding) 

        current_channels = 1
        current_t = temporal_patch_size
        current_h = spatial_patch_size
        current_w = spatial_patch_size

        self.layers = nn.ModuleList()
        
        def calculate_output_dim(input_dim, kernel_size, stride, padding):
            return (input_dim + 2 * padding - kernel_size) // stride + 1
        
        if self.layer_list is not None:
            for i, out_channels in enumerate(self.layer_list):
                self.layers.append(
                    nn.Conv3d(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                    )
                )
                
                current_channels = out_channels
                current_t = calculate_output_dim(current_t, self.kernel_size[0], self.stride[0], self.padding[0])
                current_h = calculate_output_dim(current_h, self.kernel_size[1], self.stride[1], self.padding[1])
                current_w = calculate_output_dim(current_w, self.kernel_size[2], self.stride[2], self.padding[2])
        
        self.final_layer = nn.Conv3d(
            in_channels=current_channels,
            out_channels=d_hidden,
            kernel_size=(current_t, current_h, current_w),
            stride=1,
            padding=0
        )

    def forward(self, x):
        B, NT, NS, C, T, H, W = x.shape
        x = x.view(B * NT * NS, C, T, H, W)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.final_layer(x)
        
        if self.use_final_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = x.view(B, NT * NS, self.d_hidden)
        
        return x