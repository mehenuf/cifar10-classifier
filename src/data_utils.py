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

import numpy as np

def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
):
    """
    CutMix augmentation — cuts a random patch from one image and
    pastes it into another. Labels are mixed proportionally to the
    patch area. Forces the model to learn from partial views of objects,
    dramatically reducing cat/dog and automobile/truck confusion.

    Reference: Yun et al., ICCV 2019 — 'CutMix: Regularization Strategy
    to Train Strong Classifiers With Localizable Features' (5,930 citations)

    Args:
        images: (B, C, H, W) batch tensor
        labels: (B,) integer class labels
        alpha:  Beta distribution parameter (1.0 = uniform mix ratio)

    Returns:
        mixed_images: augmented batch
        labels_a:     original labels
        labels_b:     labels of the patch source images
        lam:          mixing ratio (area of kept region)
    """
    lam    = np.random.beta(alpha, alpha)
    B, C, H, W = images.shape
    perm   = torch.randperm(B)

    # Random bounding box
    cut_w  = int(W * np.sqrt(1 - lam))
    cut_h  = int(H * np.sqrt(1 - lam))
    cx     = np.random.randint(W)
    cy     = np.random.randint(H)
    x1     = max(cx - cut_w // 2, 0)
    y1     = max(cy - cut_h // 2, 0)
    x2     = min(cx + cut_w // 2, W)
    y2     = min(cy + cut_h // 2, H)

    mixed  = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    lam    = 1 - (x2 - x1) * (y2 - y1) / (W * H)

    return mixed, labels, labels[perm], lam


def cutmix_criterion(
    criterion,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixed loss for a CutMix batch."""
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)