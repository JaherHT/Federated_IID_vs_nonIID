"""Download Fashion-MNIST and build the federated shards.

Run once (or on every machine) before any federation run, from the repo root:

    python data/prepare.py

Outputs in data/processed/ (all uint8 arrays, X (N,28,28) + y (N,)):

    test.npz                   global test set, 10000 samples (stays on the server)
    iid/c1.npz c2.npz c3.npz   IID split: one shuffle, three 20k shards
    noniid/c1.npz ...          non-IID split: sorted by label, three 20k shards
    smoke/{c1,c2,c3,test}.npz  tiny shards for the local smoke test
    distributions.json         class counts per client (the IID/non-IID evidence)
    manifest.json              seeds, source URL, checksums, dataset statistics

The splits are deterministic (fixed seed), so every machine that runs this
produces identical shards. Handing teammates their .npz files directly
(see tools/make_peer_bundle.py) is still the safest way to guarantee one
common split for the whole team.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
SEED = 42
CLIENT_IDS = ["c1", "c2", "c3"]
SMOKE_SIZE = 2500
SMOKE_TEST_SIZE = 2000


def log(message: str) -> None:
    print(f"[prepare] {message}", flush=True)


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        target = raw_dir / name
        if target.exists() and target.stat().st_size > 0:
            log(f"{name} already present ({target.stat().st_size} bytes)")
            continue
        url = BASE_URL + name
        log(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=180) as response, open(target, "wb") as handle:
            handle.write(response.read())
        log(f"  -> {target} ({target.stat().st_size} bytes)")


def load_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"{path}: unexpected magic {magic}")
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows, cols)


def load_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"{path}: unexpected magic {magic}")
    return np.frombuffer(data, dtype=np.uint8, offset=8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_shard(path: Path, X: np.ndarray, y: np.ndarray, idx: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if idx is None:
        np.savez(path, X=X.astype(np.uint8), y=y.astype(np.uint8))
    else:
        np.savez(path, X=X.astype(np.uint8), y=y.astype(np.uint8), idx=idx.astype(np.uint32))


def class_counts(y: np.ndarray) -> dict:
    counts = np.bincount(y.astype(np.int64), minlength=10)
    return {str(c): int(n) for c, n in enumerate(counts)}


def preview(y: np.ndarray) -> str:
    counts = np.bincount(y.astype(np.int64), minlength=10)
    return " ".join(f"{c}:{int(n)}" for c, n in enumerate(counts) if n > 0)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build Fashion-MNIST federated shards")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)

    download(raw_dir)

    log("parsing raw files ...")
    X_train = load_images(raw_dir / FILES["train_images"])
    y_train = load_labels(raw_dir / FILES["train_labels"])
    X_test = load_images(raw_dir / FILES["test_images"])
    y_test = load_labels(raw_dir / FILES["test_labels"])
    log(f"train: {X_train.shape}, test: {X_test.shape}")

    mean = float(X_train.astype(np.float32).mean() / 255.0)
    std = float(X_train.astype(np.float32).std() / 255.0)
    log(f"dataset stats: mean={mean:.4f} std={std:.4f} (constants live in common/data.py)")

    rng = np.random.default_rng(args.seed)

    # IID: one shuffle, then three equal chunks
    perm = rng.permutation(len(y_train))
    iid_chunks = np.array_split(perm, len(CLIENT_IDS))

    # non-IID: sort by label (the classic pathological split from the FedAvg
    # paper), then three equal chunks: each client ends up with ~3 classes
    order = np.argsort(y_train, kind="stable")
    noniid_chunks = np.array_split(order, len(CLIENT_IDS))

    distributions = {
        "iid": {},
        "noniid": {},
        "test": {"counts": class_counts(y_test), "total": int(len(y_test))},
    }

    for scenario, chunks in (("iid", iid_chunks), ("noniid", noniid_chunks)):
        log(f"--- {scenario} split ---")
        for client_id, idx in zip(CLIENT_IDS, chunks):
            X, y = X_train[idx], y_train[idx]
            save_shard(out_dir / scenario / f"{client_id}.npz", X, y, idx)
            distributions[scenario][client_id] = {"counts": class_counts(y), "total": int(len(y))}
            log(f"  {client_id}: {len(y)} samples | {preview(y)}")

    save_shard(out_dir / "test.npz", X_test, y_test)
    log(f"test.npz: {len(y_test)} samples")

    # smoke shards: small slices so the whole pipeline runs in ~1 minute
    smoke_dir = out_dir / "smoke"
    for client_id, idx in zip(CLIENT_IDS, iid_chunks):
        idx_small = idx[:SMOKE_SIZE]
        save_shard(smoke_dir / f"{client_id}.npz", X_train[idx_small], y_train[idx_small], idx_small)
    test_choice = rng.choice(len(y_test), size=min(SMOKE_TEST_SIZE, len(y_test)), replace=False)
    save_shard(smoke_dir / "test.npz", X_test[test_choice], y_test[test_choice])
    log(f"smoke shards: {SMOKE_SIZE} train / {len(test_choice)} test per client")

    distributions_path = out_dir / "distributions.json"
    distributions_path.write_text(json.dumps(distributions, indent=2), encoding="utf-8")

    manifest = {
        "seed": args.seed,
        "source": BASE_URL,
        "counts": {
            "train": int(len(y_train)),
            "test": int(len(y_test)),
            "per_client": int(len(y_train) // len(CLIENT_IDS)),
            "smoke_per_client": SMOKE_SIZE,
        },
        "dataset_stats": {"mean": round(mean, 6), "std": round(std, 6)},
        "files": {},
    }
    for rel in sorted(out_dir.rglob("*.npz")):
        manifest["files"][rel.relative_to(out_dir).as_posix()] = sha256(rel)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"distributions: {distributions_path}")
    log(f"manifest:      {manifest_path}")
    log("done.")


if __name__ == "__main__":
    main()
