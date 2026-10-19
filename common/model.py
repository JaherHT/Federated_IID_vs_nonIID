"""Client model zoo (examples).

The federation exchanges only logits, never weights, so every client may run
its own architecture. Different architectures on different machines is the
point of this setup, not a problem.

Presets (all take (1, 28, 28) input and output 10 logits):
  cnn        small 2-conv CNN, ~20k parameters
  mlp        fully connected 784-256-64-10, ~218k parameters
  cnn_deep   3-conv CNN, ~90k parameters

Add your own: write a function returning an nn.Module, add it to MODELS,
done. Only your own client uses it; nothing else in the system depends on it.
"""
from __future__ import annotations

import torch.nn as nn


def cnn(num_classes: int = 10) -> nn.Module:
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


def mlp(num_classes: int = 10) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )


def cnn_deep(num_classes: int = 10) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 3 * 3, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )


MODELS = {"cnn": cnn, "mlp": mlp, "cnn_deep": cnn_deep}


def build_model(name: str = "cnn", num_classes: int = 10) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown architecture {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](num_classes)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
