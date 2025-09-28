from typing import Tuple

import torchvision
from torch.utils.data import Dataset
from ._transform import transform

def get_CIFAR10()-> Tuple[Dataset, Dataset]:
    train = torchvision.datasets.CIFAR10(root='./.cache/data', train=True,
                                             download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root='./.cache/data', train=False,
                                            download=True, transform=transform)
    return train, test