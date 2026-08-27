"""Model definitions for the image classifier.

Supports a lightweight custom CNN and a fine-tuned torchvision ResNet-18,
selectable via the `architecture` field in the training config.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    """A small CNN baseline for CIFAR-10-sized (3x32x32) images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Return a torchvision ResNet-18 adapted for small (32x32) images.

    The stem is modified (smaller first conv, no initial maxpool) since the
    default ImageNet stem downsamples too aggressively for CIFAR-10-sized
    inputs.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(architecture: str, num_classes: int = 10) -> nn.Module:
    architecture = architecture.lower()
    if architecture in ("resnet18", "resnet-18"):
        return get_resnet18(num_classes=num_classes)
    if architecture in ("cnn", "simplecnn", "simple_cnn"):
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture}")
