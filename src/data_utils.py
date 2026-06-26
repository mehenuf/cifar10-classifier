"""Loaders, preprocessing and augmentation for CIFAR-10 dataset"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)