"""Data loading utilities.

Extends the assignment starter code with:
- Support for both CIFAR-10 and Fashion-MNIST (selectable via config)
- A shared CLASS_NAMES lookup used by both training logs and the serving API
- Configurable num_workers and download flag so this behaves correctly both
  on a laptop and inside a container with a mounted data volume
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

CLASS_NAMES = {
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
    "fashion_mnist": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ],
}


def get_transforms(dataset: str = "cifar10", train: bool = True) -> transforms.Compose:
    if dataset == "cifar10":
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])

    if dataset == "fashion_mnist":
        # Fashion-MNIST is single-channel; replicate to 3 channels so the
        # same model architectures (built for 3xHxW inputs) work unchanged.
        base = [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
        if train:
            return transforms.Compose([transforms.RandomHorizontalFlip(), *base])
        return transforms.Compose(base)

    raise ValueError(f"Unknown dataset: {dataset}")


def get_dataloaders(
    data_dir: str,
    dataset: str = "cifar10",
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    if dataset == "cifar10":
        dataset_cls = datasets.CIFAR10
    elif dataset == "fashion_mnist":
        dataset_cls = datasets.FashionMNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_dataset = dataset_cls(
        root=data_dir,
        train=True,
        download=download,
        transform=get_transforms(dataset, train=True),
    )
    val_dataset = dataset_cls(
        root=data_dir,
        train=False,
        download=download,
        transform=get_transforms(dataset, train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def get_inference_transform(dataset: str = "cifar10") -> transforms.Compose:
    """Transform applied to a single uploaded image at serving time."""
    if dataset == "cifar10":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    if dataset == "fashion_mnist":
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])
    raise ValueError(f"Unknown dataset: {dataset}")
