"""Quick CPU check: how fast is training on this machine?

    python tools/bench.py

Times one training step of the shared CNN on this CPU and extrapolates:

  * the client role pays for local training every round (the heavy part)
  * the server role only evaluates (a few seconds per round)

Run this on every laptop before the demo; if a machine is unexpectedly slow,
say so and use `training_override` (batch_size / local_epochs) in the client
config, or lower the shared settings for everyone.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.model import make_model, param_count


def main() -> None:
    torch.manual_seed(0)
    model = make_model()
    params = param_count(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))

    def step() -> None:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    for _ in range(3):  # warmup
        step()
    n_steps = 25
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    per_step = (time.perf_counter() - t0) / n_steps

    per_epoch = per_step * (20000 // 64 + 1)
    per_round = per_epoch * 1.3 + 3.0  # 1 local epoch + data/eval overhead estimate

    print(f"threads used: {torch.get_num_threads()}")
    print(f"model: {params} params ({params * 4 / 1024:.1f} KB float32 per weight upload)")
    print(f"training speed: {per_step * 1000:.1f} ms per batch of 64")
    print(f"one local epoch on a 20k shard: ~{per_epoch:.1f} s")
    print(f"one full round incl. overhead:  ~{per_round:.1f} s (estimate)")
    print(f"15 rounds on the client side:   ~{per_round * 15 / 60:.1f} min (estimate)")


if __name__ == "__main__":
    main()
