import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Union, cast

class VGG(nn.Module):
    """
    VGG architecture for image classification.
    Consists of convolutional layers followed by fully connected layers.
    """
    def __init__(self, features: nn.Module):
        super().__init__()
        self.features = features
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.feed_forward = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor(b, c, h, w)
        return: Tensor(b, 4096)
        """
        out = x
        out = self.features(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.feed_forward(out)
        return out

def make_layers(config: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    """
    Create convolutional layers using list configuration.
    cfg: List of layer specifications (int for conv channels, 'M' for max pooling)
    batch_norm: Whether to use batch normalization
    return: Sequential([layer1, layer2, ...])
    """
    layers: List[nn.Module] = []
    in_channels = 3
    for v in config:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


configurations = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11(batch_norm: bool = False) -> VGG:
    return VGG(make_layers(configurations['A'], batch_norm))

def vgg13(batch_norm: bool = False, num_classes: int = 1000) -> VGG:
    return VGG(make_layers(configurations['B'], batch_norm))

def vgg16(batch_norm: bool = False) -> VGG:
    return VGG(make_layers(configurations['D'], batch_norm))

def vgg19(batch_norm: bool = False) -> VGG:
    return VGG(make_layers(configurations['E'], batch_norm))


if __name__ == "__main__":
    # Test VGG11
    model = vgg11()
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"VGG11 output shape: {output.shape}")  # Expected: torch.Size([1, 10])

    # Test VGG16
    model = vgg16()
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"VGG16 output shape: {output.shape}")  # Expected: torch.Size([1, 10])