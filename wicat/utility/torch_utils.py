import torch.nn as nn


def get_activation_function(activation_str):
    """
    Returns activation function given the activation function's name

    Parameters:
    ----------------------
    activation_str: str, Activation function's name

    Returns:
    ----------------------
    activation_fn: Activation function from torch.nn
    """

    if activation_str.lower() == "elu":
        return nn.ELU()
    elif activation_str.lower() == "hardtanh":
        return nn.Hardtanh()
    elif activation_str.lower() == "leakyrelu":
        return nn.LeakyReLU()
    elif activation_str.lower() == "relu":
        return nn.ReLU()
    elif activation_str.lower() == "rrelu":
        return nn.RReLU()
    elif activation_str.lower() == "sigmoid":
        return nn.Sigmoid()
    elif activation_str.lower() == "mish":
        return nn.Mish()
    elif activation_str.lower() == "tanh":
        return nn.Tanh()
    elif activation_str.lower() == "tanhshrink":
        return nn.Tanhshrink()
    elif activation_str.lower() == "linear":
        return lambda x: x
    elif activation_str.lower() == "silu":
        return nn.SiLU()
    elif activation_str.lower() == "gelu":
        return nn.GELU()
