from typing import Type

import torch
from torch import nn
# from .base_components.convolution import Conv2d
from torch.nn import Conv2d

class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet.
    [3x3 conv] -> [3x3 conv] -> [Residual Conn]
    *support pre_activation*
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, pre_activation: bool = False):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels if pre_activation else out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.pre_activation = pre_activation

        self.norm_activate_layers = [
            nn.Sequential(self.bn1, self.relu),
            nn.Sequential(self.bn2, self.relu if self.pre_activation else nn.Identity()),
        ]

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x

        if self.pre_activation:
            out = self.norm_activate_layers[0](out)
            out = self.conv1(out)
            out = self.norm_activate_layers[1](out)
            out = self.conv2(out)
        else:
            out = self.conv1(out)
            out = self.norm_activate_layers[0](out)
            out = self.conv2(out)
            out = self.norm_activate_layers[1](out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if not self.pre_activation:
            out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet.
    [1x1 conv] -> [3x3 conv] -> [1x1 conv] -> [Residual Conn]
    *support pre_activation*
    """
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, pre_activation: bool = False):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels if pre_activation else out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels if pre_activation else out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.pre_activation = pre_activation

        self.norm_activate_layers = [
            nn.Sequential(self.bn1, self.relu),
            nn.Sequential(self.bn2, self.relu),
            nn.Sequential(self.bn3, self.relu if self.pre_activation else nn.Identity()),
        ]

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = x
        if self.pre_activation:
            out = self.norm_activate_layers[0](out)
            out = self.conv1(out)
            out = self.norm_activate_layers[1](out)
            out = self.conv2(out)
            out = self.norm_activate_layers[2](out)
            out = self.conv3(out)
        else:
            out = self.conv1(out)
            out = self.norm_activate_layers[0](out)
            out = self.conv2(out)
            out = self.norm_activate_layers[1](out)
            out = self.conv3(out)
            out = self.norm_activate_layers[2](out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if not self.pre_activation:
            out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Residual Net
    """
    def __init__(self, block: Type[nn.Module], layers: list[int], pre_activation: bool = False):
        super().__init__()
        self.in_channels = 64
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.Sequential(
            self._make_layer(block, 64, layers[0], pre_activation=pre_activation),
            self._make_layer(block, 128, layers[1], stride=2, pre_activation=pre_activation),
            self._make_layer(block, 256, layers[2], stride=2, pre_activation=pre_activation),
            self._make_layer(block, 512, layers[3], stride=2, pre_activation=pre_activation)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self,
                    block: Type[nn.Module],
                    out_channels: int,
                    blocks: int,
                    stride: int = 1,
                    pre_activation: bool = False) -> nn.Sequential:
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, pre_activation=pre_activation))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, pre_activation=pre_activation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layers(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        return x


def resnet18(pre_activation: bool = False) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], pre_activation=pre_activation)

def resnet34(pre_activation: bool = False) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], pre_activation=pre_activation)

def resnet50(pre_activation: bool = False) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], pre_activation=pre_activation)

def resnet101(pre_activation: bool = False) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], pre_activation=pre_activation)

def resnet152(pre_activation: bool = False) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], pre_activation=pre_activation)

if __name__ == '__main__':
    def test_resnet18():
        model = resnet18()
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        print("ResNet18 output shape:", y.shape)  # Expected: torch.Size([1, 10])

    def test_resnet50():
        model = resnet50()
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        print("ResNet50 output shape:", y.shape)  # Expected: torch.Size([1, 10])

    test_resnet18()
    test_resnet50()