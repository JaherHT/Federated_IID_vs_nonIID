"""Protocol conformance check for client implementations.

Self-check with the reference client (no other machine needed):

    python tools/protocol_selftest.py --mode all

Check YOUR OWN client implementation:

    python tools/protocol_selftest.py --mode server --port 5001
    # then point your client at 127.0.0.1:5001 (any shard file works;
    # data/processed/smoke/c1.npz is the small one) and wait for PASS/FAIL

Prove that a RUNNING server refuses anything that is not this activity
(while the server is up, e.g. while it waits for clients):

    python tools/protocol_selftest.py --mode probe --port 5000

The test server does no federation: it runs three rounds (two logits rounds,
then one FedAvg weights round), echoes the client's first probabilities back
as the consensus for the second round, and validates every message against
PROTOCOL.md (HELLO fields, exactly one UPDATE per round, probability
shape/dtype/range in logits rounds, delta keys/shapes in the FedAvg round,
finite values, n_train > 0).

Requires: secret.key (tools/gen_key.py) and the smoke files
(data/prepare.py) for --mode all.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import protocol
from common.model import build_model

DEFAULT_PORT = 5001
SMOKE_PUBLIC = "data/processed/smoke/public.npz"


def _check_summary(probs: np.ndarray, n_public: int, n_classes: int = 10) -> str:
    if not isinstance(probs, np.ndarray):
        return "missing probability array"
    if probs.dtype != np.float32:
        return f"dtype {probs.dtype}, expected float32"
    if probs.shape != (n_public, n_classes):
        return f"shape {tuple(probs.shape)}, expected ({n_public}, {n_classes})"
    if not bool(np.isfinite(probs).all()):
        return "non-finite values"
    if bool((probs < -1e-6).any()) or bool((probs > 1.0 + 1e-6).any()):
        return "values outside [0, 1]"
    sums = probs.sum(axis=1)
    if bool((sums < 0.9).any()) or bool((sums > 1.1).any()):
        return "rows do not sum to ~1"
    return ""


def run_test_server(port: int, key: bytes, results: dict, timeout_s: float = 240.0) -> None:
    import json

    checks = results["checks"] = []
    listener = None
    conn = None

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    n_public = 0
    try:
        with np.load(SMOKE_PUBLIC) as payload:
            n_public = int(payload["X"].shape[0])

        listener = protocol.make_listener("127.0.0.1", port)
        listener.settimeout(timeout_s)
        sock, addr = listener.accept()
        sock.settimeout(timeout_s)
        conn = protocol.from_socket(sock, key)

        hello = conn.recv()
        check("first message is HELLO", hello.get("type") == "HELLO", repr(hello.get("type")))
        check(f"HELLO.proto == {protocol.PROTO_VERSION}",
              int(hello.get("proto", -1)) == protocol.PROTO_VERSION)
        check("HELLO.client_id present", bool(hello.get("client_id")), repr(hello.get("client_id")))
        check("HELLO.n_train > 0", int(hello.get("n_train", 0)) > 0, str(hello.get("n_train")))
        check("HELLO.ready is true", hello.get("ready") is True, repr(hello.get("ready")))
        check("HELLO.arch present", bool(hello.get("arch")), repr(hello.get("arch")))

        conn.send({"type": "WELCOME", "proto": protocol.PROTO_VERSION, "total_rounds": 3,
                   "public_size": n_public, "num_classes": 10, "strategy": "logits",
                   "shared_model": "cnn"})
        conn.send({"type": "ROUND", "round": 1, "targets": None,
                   "config": {"local_epochs": 1, "kd_epochs": 0, "temperature": 2.0,
                              "batch_size": 64, "lr": 0.001, "optimizer": "adam"}})

        update = conn.recv()
        check("round 1: message is UPDATE", update.get("type") == "UPDATE", repr(update.get("type")))
        check("round 1: UPDATE.round == 1", int(update.get("round", -1)) == 1, str(update.get("round")))
        check("round 1: UPDATE.n_train > 0", int(update.get("n_train", 0)) > 0, str(update.get("n_train")))
        problem = _check_summary(update.get("probs"), n_public)
        check("round 1: probabilities valid", not problem, problem)
        aux = update.get("aux")
        check("round 1: aux is a dict of scalars", isinstance(aux, dict) and
              all(isinstance(v, (int, float, bool)) for v in aux.values()))
        first_probs = update.get("probs") if isinstance(update.get("probs"), np.ndarray) else None

        # second round: echo their probabilities back as the consensus; the
        # client must distill from it and produce another valid UPDATE
        targets = first_probs if first_probs is not None else np.full((n_public, 10), 0.1, dtype=np.float32)
        conn.send({"type": "ROUND", "round": 2, "targets": targets,
                   "config": {"local_epochs": 1, "kd_epochs": 1, "temperature": 2.0,
                              "batch_size": 64, "lr": 0.001, "optimizer": "adam"}})
        update2 = conn.recv()
        check("round 2: message is UPDATE", update2.get("type") == "UPDATE", repr(update2.get("type")))
        check("round 2: UPDATE.round == 2", int(update2.get("round", -1)) == 2, str(update2.get("round")))
        problem2 = _check_summary(update2.get("probs"), n_public)
        check("round 2: probabilities valid after distillation", not problem2, problem2)

        # third round: FedAvg weights exchange - the client must return a delta
        weights = {name: tensor.detach().cpu().numpy().astype("float32")
                   for name, tensor in build_model("cnn").state_dict().items()}
        conn.send({"type": "ROUND", "round": 3, "weights": weights,
                   "config": {"local_epochs": 1, "kd_epochs": 0, "temperature": 2.0,
                              "batch_size": 64, "lr": 0.001, "optimizer": "adam"}})
        update3 = conn.recv()
        check("round 3: message is UPDATE", update3.get("type") == "UPDATE", repr(update3.get("type")))
        check("round 3: UPDATE.round == 3", int(update3.get("round", -1)) == 3, str(update3.get("round")))
        delta = update3.get("delta")
        same_keys = isinstance(delta, dict) and set(delta.keys()) == set(weights.keys())
        check("round 3 (FedAvg): delta keys match the shared model", same_keys)
        if same_keys:
            check("round 3 (FedAvg): delta shapes match",
                  all(tuple(np.asarray(delta[k]).shape) == tuple(weights[k].shape) for k in weights))
            check("round 3 (FedAvg): delta values are finite",
                  all(bool(np.isfinite(np.asarray(delta[k])).all()) for k in weights))

        try:
            conn.send({"type": "FINISHED"})
        except Exception:
            pass
        results["done"] = True
    except Exception as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            conn.close()
        if listener is not None:
            listener.close()


def report(results: dict) -> int:
    checks = results.get("checks", [])
    ok = bool(checks) and all(c[1] for c in checks) and bool(results.get("done"))
    for name, passed, detail in checks:
        line = f"  [{'PASS' if passed else 'FAIL'}] {name}"
        if not passed and detail:
            line += f"  (got: {detail})"
        print(line)
    if "error" in results:
        print(f"  [FAIL] test server error: {results['error']}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def mode_server(port: int) -> int:
    key = protocol.load_key("secret.key")
    results: dict = {}
    print(f"[selftest] two-round test server on 127.0.0.1:{port}; waiting for a client ...")
    thread = threading.Thread(target=run_test_server, args=(port, key, results), daemon=True)
    thread.start()
    thread.join()
    return report(results)


def mode_all() -> int:
    shard = Path("data/processed/smoke/c1.npz")
    public = Path(SMOKE_PUBLIC)
    if not shard.exists() or not public.exists():
        print("[selftest] missing smoke files; run data/prepare.py first")
        return 2
    if not Path("secret.key").exists():
        print("[selftest] missing secret.key; run tools/gen_key.py first")
        return 2
    key = protocol.load_key("secret.key")
    port = DEFAULT_PORT + 1
    results: dict = {}
    server = threading.Thread(target=run_test_server, args=(port, key, results), daemon=True)
    server.start()

    from client.client import Client  # the reference client

    cfg = {
        "server_host": "127.0.0.1",
        "server_port": port,
        "client_id": "selftest",
        "architecture": "cnn",
        "shard_file": str(shard),
        "public_file": str(public),
        "connect_timeout_s": 5,
        "retry_delay_s": 2,
        "local_metrics": True,
        "encryption": {"enabled": True, "key_file": "secret.key"},
    }
    client_thread = threading.Thread(target=lambda: Client(cfg).run(), daemon=True)
    client_thread.start()
    server.join()
    client_thread.join(timeout=5)
    return report(results)


def mode_probe(port: int) -> int:
    """Prove a RUNNING server refuses everything that is not the activity."""
    key = protocol.load_key("secret.key") if Path("secret.key").exists() else None
    checks = []

    def probe(name: str, payload: dict, expect: str) -> None:
        try:
            conn = protocol.connect("127.0.0.1", port, key, 5.0)
        except Exception as exc:
            checks.append((name, False, f"could not connect: {exc}"))
            return
        try:
            conn.send(payload)
            try:
                reply = conn.recv()
                checks.append((name, False, f"server answered {reply.get('type')!r} instead of closing"))
            except Exception:
                checks.append((name, True, expect))
        finally:
            conn.close()

    probe("junk message as first contact is dropped",
          {"type": "EXFILTRATE", "path": "C:/Users", "cmd": "run"},
          "connection closed by server")
    probe("unknown client_id is refused",
          {"type": "HELLO", "proto": protocol.PROTO_VERSION, "client_id": "intruder",
           "n_train": 1000, "ready": True, "arch": "x"},
          "connection closed by server")
    probe("client without ready flag is refused",
          {"type": "HELLO", "proto": protocol.PROTO_VERSION, "client_id": "c1",
           "n_train": 1000, "arch": "x"},
          "connection closed by server")

    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if not ok else ""))
    all_ok = all(ok for _, ok, _ in checks)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Protocol conformance check")
    parser.add_argument("--mode", choices=["all", "server", "probe"], default="all")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    if args.mode == "all":
        code = mode_all()
    elif args.mode == "server":
        code = mode_server(args.port)
    else:
        code = mode_probe(args.port)
    sys.exit(code)


if __name__ == "__main__":
    main()
