"""Create one ready-to-send zip per teammate: repo copy + their shards + key.

    python tools/make_peer_bundle.py                  # bundles for c1, c2, c3
    python tools/make_peer_bundle.py --peers c1 c2
    python tools/make_peer_bundle.py --scenario iid   # only the IID shards

Each zip (bundles/peer_<id>.zip) contains:

    the code (common/, server/, client/, tools/, docs, requirements.txt)
    data/processed/<scenario>/<id>.npz   - that peer's shard(s), read-only
    data/processed/public.npz            - the shared label-free public set
    client/config.json                   - pre-filled for that peer
    secret.key                           - the shared protocol key
    START_HERE.txt                       - exact setup and run commands

Send each zip privately (it contains the key). Never commit bundles/ to git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

INCLUDE_DIRS = ["common", "server", "client", "tools"]
INCLUDE_FILES = ["README.md", "PROTOCOL.md", "TEAM.md", "requirements.txt", ".gitignore"]

# One architecture per client on purpose: each machine runs a different model.
ARCHITECTURES = {"c1": "cnn", "c2": "mlp", "c3": "cnn_deep"}

START_HERE = """\
START HERE - client '{client_id}'
=================================

You are client '{client_id}' of a 3-client federation (Fashion-MNIST,
every client runs its own model). One client process, from this folder:

  1. python -m venv .venv
     .venv\\Scripts\\pip install -r requirements.txt
     (CPU-only torch, smaller download:
      .venv\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cpu)

  2. Self-check; expect RESULT: PASS:
     .venv\\Scripts\\python tools\\bench.py
     .venv\\Scripts\\python tools\\protocol_selftest.py --mode all

  3. Run: set "server_host" in client\\config.json (address from the group
     chat; see TEAM.md for same-network vs different-network addresses), then:
     .venv\\Scripts\\python -m client.client --config client\\config.json
     Keep it running; it reconnects by itself after network drops.

Your file:  client\\config.json  ->  "architecture": "{arch}"
  Presets: cnn | mlp | cnn_deep (common\\model.py). Swap freely, or add your
  own model function there; different machines may run different models.

Your shards (read-only):
     data\\processed\\iid\\{client_id}.npz       (IID run)
     data\\processed\\noniid\\{client_id}.npz    (non-IID run)
Switch runs by pointing shard_file in client\\config.json at the other file.

secret.key in this folder is the shared key; keep it private.
Details: TEAM.md (roles, duties, steps) and PROTOCOL.md (wire format).
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_code_files() -> list:
    files = []
    for folder in INCLUDE_DIRS:
        base = REPO_ROOT / folder
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files.append(path.relative_to(REPO_ROOT))
    for name in INCLUDE_FILES:
        path = REPO_ROOT / name
        if path.exists():
            files.append(Path(name))
    return files


def build_peer(peer: str, scenarios: list, out_dir: Path) -> Path:
    zip_path = out_dir / f"peer_{peer}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in collect_code_files():
            zf.write(REPO_ROOT / rel, arcname=rel.as_posix())
        for scenario in scenarios:
            shard = REPO_ROOT / "data" / "processed" / scenario / f"{peer}.npz"
            if not shard.exists():
                raise SystemExit(f"missing shard {shard}; run data/prepare.py first")
            zf.write(shard, arcname=f"data/processed/{scenario}/{peer}.npz")
        public = REPO_ROOT / "data" / "processed" / "public.npz"
        if not public.exists():
            raise SystemExit("missing data/processed/public.npz; run data/prepare.py first")
        zf.write(public, arcname="data/processed/public.npz")
        zf.writestr("START_HERE.txt", START_HERE.format(client_id=peer, arch=ARCHITECTURES.get(peer, "cnn")))
        config = {
            "server_host": "PUT-SERVER-IP-HERE",
            "server_port": 5000,
            "client_id": peer,
            "architecture": ARCHITECTURES.get(peer, "cnn"),
            "shard_file": f"data/processed/{scenarios[0]}/{peer}.npz",
            "public_file": "data/processed/public.npz",
            "encryption": {"enabled": True, "key_file": "secret.key"},
            "connect_timeout_s": 10,
            "retry_delay_s": 3,
            "local_metrics": True,
            "training_override": None,
            "log_file": None,
        }
        zf.writestr("client/config.json", json.dumps(config, indent=2))
        key_path = REPO_ROOT / "secret.key"
        if not key_path.exists():
            raise SystemExit("missing secret.key; run tools/gen_key.py first")
        zf.write(key_path, arcname="secret.key")
    return zip_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build per-peer zip bundles")
    parser.add_argument("--peers", nargs="+", default=["c1", "c2", "c3"])
    parser.add_argument("--scenario", choices=["iid", "noniid", "both"], default="both")
    parser.add_argument("--out", default="bundles")
    args = parser.parse_args(argv)

    scenarios = ["iid", "noniid"] if args.scenario == "both" else [args.scenario]
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for peer in args.peers:
        zip_path = build_peer(peer, scenarios, out_dir)
        size_mb = zip_path.stat().st_size / 1e6
        print(f"{zip_path}  ({size_mb:.1f} MB)  sha256={sha256(zip_path)[:16]}...")
    print("send each zip privately (it contains secret.key); do not commit bundles/ to git.")


if __name__ == "__main__":
    main()
