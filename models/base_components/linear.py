from abc import abstractmethod

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter


class Linear(nn.Module):
    """
    out = x@weight.T [+ bias]

    input_dim: feature dimension of the input, aka the size of last dimension
    output_dim: output size of the linear layer
    """
    def __init__(self, input_dim: int, output_dim: int, use_bias=True):
        super().__init__()
        # Register weights: Tensor(out_d, in_d)
        self.weight = Parameter(torch.randn(output_dim, input_dim))

        # Register bias: Tensor(out_d)
        if use_bias:
            self.bias = Parameter(torch.randn(output_dim))
        else:
            self.register_parameter("bias", None)

        # Init parameters
        self.init_parameters()

    def init_parameters(self) -> None:
        pass

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor(any, in_d)
        return: Tensor(any, out_d)
        """
        if self.bias is not None:
            out = x.matmul(self.weight.t()) + self.bias
        else:
            out = x.matmul(self.weight.t())
        return out


class DropoutBase(nn.Module):
    """
    base class for dropout layers
    """
    def __init__(self, percentage: float, inplace: bool = False):
        super().__init__()
        self.percentage = percentage
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        """
        Mask the input, then scale the rest to 1/(1-percentage).
        x: Tensor(any)
        return: Tensor(any)
        """
        if not self.training:
            return x
        else:
            mask = self.generate_dropout_mask_()
            if self.inplace:
                x.mul_(mask)
                return x
            else:
                return x * mask

    @abstractmethod
    def generate_dropout_mask_(self, x: Tensor) -> Tensor:
        raise NotImplementedError

class DropOutPixel(DropoutBase):
    def generate_dropout_mask_(self, x: Tensor) -> Tensor:
        """
        Return a mask for each pixel.
        Shape: x.shape
        """
        return torch.rand_like(x) < self.percentage


class DropOut1D(DropoutBase):
    """
    Channel wise dropout for in 1d feature input.
    Examples: text embedding(batch_size, segment_len, feature_size)
    """
    def generate_dropout_mask_(self, x: Tensor) -> Tensor:
        """
        Return a mask for
        Shape: (x.shape[:-1], 1)
        """
        shape = list(x.shape[:-1]) + [1]
        return torch.rand(shape) < self.percentage


class DropOut2D(DropoutBase):
    """
    Channel wise dropout for in 2d feature input.
    Examples: conv layer output(batch_size, channels, height, width)
    """
    def generate_dropout_mask_(self, x: Tensor) -> Tensor:
        """
        Shape: (x.shape[:-2], 1, 1)
        """
        shape = list(x.shape[:-1]) + [1, 1]
        return torch.rand(shape) < self.percentage

class DropOut3D(DropoutBase):
    """
    Channel wise dropout for in 3d feature input.
    Examples: conv layer output(batch_size, channels, seq_len, height, width)
    """
    def generate_dropout_mask_(self, x: Tensor) -> Tensor:
        """
        Shape: (x.shape[:-3], 1, 1, 1)
        """
        shape = list(x.shape[:-3]) + [1, 1, 1]
        return torch.rand(shape) < self.percentage