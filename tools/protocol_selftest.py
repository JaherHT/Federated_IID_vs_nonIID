"""Protocol conformance check for client implementations.

Self-check with the reference client (no other machine needed):

    python tools/protocol_selftest.py --mode all

Check YOUR OWN client implementation:

    python tools/protocol_selftest.py --mode server --port 5001
    # then point your client at 127.0.0.1:5001 (any shard file works;
    # data/processed/smoke/c1.npz is the small one) and wait for PASS/FAIL

The test server does no federation: it runs exactly one round with random
weights and validates the client's messages against PROTOCOL.md: HELLO
fields, exactly one UPDATE for the right round, delta keys and shapes
matching the server weights, finite values, n_train > 0.

Requires: secret.key (tools/gen_key.py) and the smoke shards
(data/prepare.py) for --mode all.
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import protocol
from common.model import make_model

DEFAULT_PORT = 5001


def run_test_server(port: int, key: bytes, results: dict, timeout_s: float = 240.0) -> None:
    import numpy as np

    checks = results["checks"] = []
    listener = None
    conn = None

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    try:
        listener = protocol.make_listener("127.0.0.1", port)
        listener.settimeout(timeout_s)
        sock, addr = listener.accept()
        sock.settimeout(timeout_s)
        conn = protocol.from_socket(sock, key)

        hello = conn.recv()
        check("first message is HELLO", hello.get("type") == "HELLO", repr(hello.get("type")))
        check("HELLO.proto == 1", int(hello.get("proto", -1)) == protocol.PROTO_VERSION)
        check("HELLO.client_id present", bool(hello.get("client_id")), repr(hello.get("client_id")))
        check("HELLO.n_train > 0", int(hello.get("n_train", 0)) > 0, str(hello.get("n_train")))

        conn.send({"type": "WELCOME", "proto": protocol.PROTO_VERSION, "total_rounds": 1})
        weights = {k: v.detach().cpu().numpy().astype("float32") for k, v in make_model().state_dict().items()}
        conn.send({"type": "ROUND", "round": 1, "weights": weights,
                   "config": {"local_epochs": 1, "batch_size": 64, "lr": 0.001, "optimizer": "adam"}})

        update = conn.recv()
        check("second message is UPDATE", update.get("type") == "UPDATE", repr(update.get("type")))
        check("UPDATE.round == 1", int(update.get("round", -1)) == 1, str(update.get("round")))
        check("UPDATE.n_train > 0", int(update.get("n_train", 0)) > 0, str(update.get("n_train")))
        delta = update.get("delta")
        same_keys = isinstance(delta, dict) and set(delta.keys()) == set(weights.keys())
        check("delta keys match the server weights", same_keys)
        if same_keys:
            check("delta shapes match",
                  all(tuple(np.asarray(delta[k]).shape) == tuple(weights[k].shape) for k in weights))
            check("delta values are finite",
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
    print(f"[selftest] one-round test server on 127.0.0.1:{port}; waiting for a client ...")
    thread = threading.Thread(target=run_test_server, args=(port, key, results), daemon=True)
    thread.start()
    thread.join()
    return report(results)


def mode_all() -> int:
    shard = Path("data/processed/smoke/c1.npz")
    if not shard.exists():
        print("[selftest] missing data/processed/smoke/c1.npz; run data/prepare.py first")
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
        "shard_file": str(shard),
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Protocol conformance check")
    parser.add_argument("--mode", choices=["all", "server"], default="all")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    sys.exit(mode_all() if args.mode == "all" else mode_server(args.port))


if __name__ == "__main__":
    main()
