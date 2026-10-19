"""Quick timing check: how long one federation round takes on this machine.

    python tools/bench.py                # default architecture (cnn)
    python tools/bench.py --model mlp
    python tools/bench.py --model all    # every preset, one after another

Times the real per-round client work at full size: one local epoch on a 20k
private shard, the distillation pass over the public set (2000 samples), the
probabilities pass, and the local accuracy evaluation. Run it once per
machine before exam day so nobody is surprised by round times.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from common import data as data_mod
from common.model import MODELS, build_model, param_count


def time_local_epoch(model, shard_path: str, batch_size: int, lr: float) -> tuple:
    loader = data_mod.make_loader(shard_path, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    started = time.time()
    total, batches = 0.0, 0
    for x, y in loader:
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        batches += 1
    return time.time() - started, batches, total / max(batches, 1)


def time_distill(model, public_X, temperature: float, batch_size: int, lr: float, epochs: int) -> float:
    targets = torch.full((public_X.shape[0], 10), 0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    started = time.time()
    n = int(public_X.shape[0])
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logp = F.log_softmax(model(public_X[idx]) / temperature, dim=1)
            loss = F.kl_div(logp, targets[idx], reduction="batchmean") * (temperature * temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return time.time() - started


@torch.no_grad()
def time_probs(model, public_X, batch_size: int = 512) -> float:
    model.eval()
    started = time.time()
    for i in range(0, int(public_X.shape[0]), batch_size):
        torch.softmax(model(public_X[i:i + batch_size]) / 2.0, dim=1)
    return time.time() - started


def bench(name: str, shard_path: str, public_path: str, batch_size: int, lr: float) -> None:
    model = build_model(name)
    with np.load(public_path) as payload:
        public_X = data_mod.normalize_images(payload["X"].copy())
    print(f"--- {name} ({param_count(model):,} parameters) ---")
    t_epoch, batches, loss = time_local_epoch(model, shard_path, batch_size, lr)
    print(f"  local epoch on {shard_path}: {t_epoch:.1f} s ({batches} batches, train loss {loss:.4f})")
    t_distill = time_distill(model, public_X, 2.0, batch_size, lr, epochs=2)
    print(f"  distillation (2 epochs on {int(public_X.shape[0])} public samples): {t_distill:.1f} s")
    t_probs = time_probs(model, public_X)
    print(f"  probabilities pass: {t_probs:.1f} s")
    t_eval_started = time.time()
    data_mod.evaluate(model, shard_path)
    t_eval = time.time() - t_eval_started
    print(f"  local accuracy pass: {t_eval:.1f} s")
    total = t_epoch + t_distill + t_probs + t_eval
    print(f"  => one round of client work here: ~{total:.1f} s (plus network)\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Per-round timing check")
    parser.add_argument("--model", default="cnn", help=f"one of {sorted(MODELS)} or 'all'")
    parser.add_argument("--shard", default="data/processed/iid/c1.npz")
    parser.add_argument("--public", default="data/processed/public.npz")
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args(argv)

    if not Path(args.shard).exists():
        raise SystemExit(f"missing {args.shard}; run data/prepare.py first")
    names = sorted(MODELS) if args.model == "all" else [args.model]
    for name in names:
        bench(name, args.shard, args.public, args.batch_size, args.lr)
    print("done; round wall time is the slowest client's number, not the sum.")


if __name__ == "__main__":
    main()
