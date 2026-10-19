"""Shared data helpers: shard loading, normalization, loaders, evaluation."""
from __future__ import annotations

import numpy as np
import torch

# Fashion-MNIST normalization constants (computed once in data/prepare.py from
# the training set, then fixed here so every machine uses identical values).
MEAN = 0.2860
STD = 0.3530
DEFAULT_BATCH_SIZE = 64


class Shard:
    """One machine's dataset slice: .npz with uint8 arrays X (N, 28, 28) and y (N,).

    Same class reads a client shard or the global test file; the format is
    identical on purpose.
    """

    def __init__(self, path: str):
        with np.load(path) as payload:
            self.X = payload["X"].copy()
            self.y = payload["y"].copy()
        self.path = str(path)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        x = torch.from_numpy(np.asarray(self.X[index], dtype=np.float32)).unsqueeze(0)
        x = (x / 255.0 - MEAN) / STD
        return x, int(self.y[index])

    def class_counts(self) -> dict:
        counts = np.bincount(self.y.astype(np.int64), minlength=10)
        return {int(c): int(n) for c, n in enumerate(counts)}


def make_loader(path: str, batch_size: int = DEFAULT_BATCH_SIZE, shuffle: bool = True):
    from torch.utils.data import DataLoader

    return DataLoader(Shard(path), batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate(model, path: str, batch_size: int = 512) -> tuple:
    """Mean cross-entropy loss and accuracy on a shard or the test file.

    Returns (loss, accuracy). Restores the model to train mode afterwards.
    """
    model.eval()
    loader = make_loader(path, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        logits = model(x)
        total_loss += float(criterion(logits, y))
        correct += int((logits.argmax(dim=1) == y).sum())
        total += int(y.shape[0])
    model.train()
    return total_loss / max(total, 1), correct / max(total, 1)
