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

def get_train_transforms(augment=True):
    """
    Training pipeline.
    augment=True  → RandomCrop + HFlip + ColorJitter + RandomErasing
    augment=False → normalisation only (for ablation studies)
    """
    if not augment:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def get_test_transforms():
    """Minimal pipeline for validation and test — no augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_dataloaders(
    data_dir="./data",
    batch_size=128,
    val_split=0.1,
    num_workers=0, #making num_workers=0 to avoid issues on Windows since i have faced similar issues before while using jupyter notebook 
    augment=True,
    seed=42,
):
    train_tf = get_train_transforms(augment=augment)
    test_tf  = get_test_transforms()

    # Download dataset (only happens once)
    full_train = datasets.CIFAR10(data_dir, train=True,
                                  download=True, transform=train_tf)
    full_val   = datasets.CIFAR10(data_dir, train=True,
                                  download=True, transform=test_tf)
    test_set   = datasets.CIFAR10(data_dir, train=False,
                                  download=True, transform=test_tf)

    # Deterministic train / val split
    n_total = len(full_train)          # 50,000
    n_val   = int(n_total * val_split) # 5,000
    n_train = n_total - n_val          # 45,000

    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(
        range(n_total), [n_train, n_val], generator=gen
    )

    train_loader = DataLoader(
        Subset(full_train, train_idx.indices),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_val, val_idx.indices),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[data] train={len(train_idx):,} | val={len(val_idx):,} | test={len(test_set):,}")
    return train_loader, val_loader, test_loader


def denormalise(tensor):
    """Reverse CIFAR-10 normalisation for visualisation."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)