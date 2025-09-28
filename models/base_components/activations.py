from typing import Mapping, Any

import torch
from torch import nn


class Sigmoid(nn.Module):
    """
    Sigmoid(x) = 1 / (1 + exp(-x))
               = exp(x) / (1 + exp(x))
    f(x) belong to (0, 1)
    f_prime(x) belong to (0, 1/4)
    """
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        """
        Fix overflow: Sigmoid(x) = exp(x) / (1 + exp(x)) when x < 0
        """
        return torch.where(
            x < 0,
            torch.exp(x) / (1 + torch.exp(x)),
            1 / (1 + torch.exp(-x))
        )


class Tanh(nn.Module):
    """
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    f(x) belong to (-1, 1)
    f_prime(x) belong to (0, 1)
    """
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        """
        Fix overflow: Tanh(x) = x / |x| when |x| > 20
        """
        return torch.where(
            x > 20,
            x / torch.abs(x),
            (torch.exp(x) -  torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        )


class ReLU(nn.Module):
    """
    ReLU(x) = max(0, x)
    """
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return torch.where(x <= 0, 0, x)


class LeakyReLU(nn.Module):
    """
    LeakyReLU(x) = max(negative_slope * x, x)
    """
    def __init__(self, negative_slope: float = 1e-2):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.where(x <= 0, self.negative_slope * x, torch.relu(x))

    def extra_repr(self):
        return f'negative_slope={self.negative_slope}'


class Softmax(nn.Module):
    """
    Softmax(x) = exp(x) / sum(exp(x))
    """
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        """
        Fix overflow: Softmax(x) = Softmax(x - x_max)
        """
        x -= torch.max(x)
        return torch.exp(x) / torch.sum(torch.exp(x))