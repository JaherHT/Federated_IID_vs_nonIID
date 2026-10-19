"""The one model topology used by the whole team.

FedAvg averages parameter tensors elementwise, so the server and every client
must run exactly this architecture with exactly these parameter names.

To change the model: change it here, for everyone, at once (one team-wide
edit). Never change it on a single machine only.

Parameter count: 20,490 (~80 KB as float32), so one full model fits in a
single small TCP message and round trips stay fast on a plain laptop CPU.
"""
from __future__ import annotations

import torch.nn as nn


def make_model(num_classes: int = 10) -> nn.Module:
    """Small CNN for 28x28 grayscale images (Fashion-MNIST, 10 classes).

    Layout: conv(1->16) -> relu -> maxpool -> conv(16->32) -> relu -> maxpool
            -> flatten -> linear(1568->10)
    """
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, num_classes),
    )


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
