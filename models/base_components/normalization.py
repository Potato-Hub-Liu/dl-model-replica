from abc import abstractmethod

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter


class BatchNormBase(nn.Module):
    '''
    Base class for batch normalization layers.
    '''
    def __init__(self, num_features: int, epsilon: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        '''
        Apply batch normalization.
        x: Tensor(any)
        return: Tensor(any)
        '''
        if self.training:
            # Calculate batch statistics
            batch_mean = x.mean(dim=self._get_dims(x))
            batch_var = x.var(dim=self._get_dims(x), unbiased=False)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var * (
                        x.shape[0] / (x.shape[0] - 1))  # Unbiased variance
            self.num_batches_tracked += 1

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Reshape for broadcasting
        shape = [1, -1] + [1] * (x.dim() - 2)
        mean = mean.view(shape)
        var = var.view(shape)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.epsilon)

        # Apply affine transformation
        if self.affine:
            gamma = self.gamma.view(shape)
            beta = self.beta.view(shape)
            return gamma * x_normalized + beta
        else:
            return x_normalized

    @abstractmethod
    def _get_dims(self, x: Tensor):
        raise NotImplementedError


class BatchNorm1d(BatchNormBase):
    '''
    Batch normalization for 1D input.
    '''
    def _get_dims(self, x: Tensor):
        return [0] + list(range(2, x.dim()))


class BatchNorm2d(BatchNormBase):
    '''
    Batch normalization for 2D input (e.g., conv layer output).
    '''
    def _get_dims(self, x: Tensor):
        return [0, 2, 3]


class BatchNorm3d(BatchNormBase):
    '''
    Batch normalization for 3D input (e.g., conv3d layer output).
    '''
    def _get_dims(self, x: Tensor):
        return [0, 2, 3, 4]


class LayerNorm(nn.Module):
    '''
    Layer normalization.
    '''
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
            self.beta = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: Tensor) -> Tensor:
        '''
        Apply layer normalization.
        x: Tensor(any)
        return: Tensor(any)
        '''
        # Calculate mean and variance over the last D dimensions, where D is the length of normalized_shape
        dims = tuple(range(x.dim() - len(self.normalized_shape), x.dim()))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Apply element-wise affine transformation
        if self.elementwise_affine:
            return self.gamma * x_normalized + self.beta
        else:
            return x_normalized


if __name__ == '__main__':
        print('BN:')
        x = torch.randn(4, 3, 10, 10) # (N, C, H, W)
        bn = BatchNorm2d(3)
        bn.train()

        y = bn(x)
        print(f'Input shape: {x.shape}')
        print(f'Output shape: {y.shape}')

        mean = y.mean(dim=[0, 2, 3])
        var = y.var(dim=[0, 2, 3], unbiased=False)
        print(f'OutMean: {mean.mean()}')
        print(f'OutVar: {var.mean()}')

        print('LN:')
        ln = LayerNorm([3, 10, 10])
        ln.train()

        y = ln(x)
        print(f'Input: {x.shape}')
        print(f'Output: {y.shape}')

        # Check mean and variance for each item in the batch
        mean = y.mean(dim=[1, 2, 3])
        var = y.var(dim=[1, 2, 3], unbiased=False)
        print(f'OutMean: {mean.mean()}')
        print(f'OutVar: {var.mean()}')