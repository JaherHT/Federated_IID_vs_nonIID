"""End-to-end local smoke test: one server + two clients on this machine.

    python tools/smoke_local.py                  # 3 rounds, plain run
    python tools/smoke_local.py --drop c2        # kill a client mid-run, restart it later
    python tools/smoke_local.py --slow c2        # client freezes one round (server must time it out)
    python tools/smoke_local.py --dp --rounds 3  # clipping + noise path on the consensus

Launches the real server and real clients as separate processes (small smoke
files, so a run takes about a minute), watches the server log, and prints a
PASS/FAIL checklist. With --drop / --slow it also reproduces the exam's
disconnection test locally.

Requires: secret.key (tools/gen_key.py) and smoke files (data/prepare.py).
Logs and outputs land in runs/smoke_<timestamp>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = {"c1": "cnn", "c2": "mlp"}  # two different models on purpose


def wait_for(path: Path, pattern: str, timeout_s: float) -> bool:
    regex = re.compile(pattern)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if regex.search(text):
            return True
        time.sleep(0.5)
    return False


def spawn(args: list, log_path: Path) -> subprocess.Popen:
    handle = open(log_path, "wb")
    return subprocess.Popen([sys.executable, *args], cwd=str(REPO_ROOT),
                            stdout=handle, stderr=subprocess.STDOUT)


def stop(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def wait_exit(proc, timeout_s: float = 15.0):
    if proc is None:
        return None
    deadline = time.time() + timeout_s
    while time.time() < deadline and proc.poll() is None:
        time.sleep(0.5)
    return proc.poll()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Local end-to-end smoke test")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--port", type=int, default=5099)
    parser.add_argument("--round-timeout", type=float, default=25.0, dest="round_timeout")
    parser.add_argument("--drop", default=None, metavar="CLIENT_ID",
                        help="kill this client mid-run and restart it a few seconds later")
    parser.add_argument("--slow", default=None, metavar="CLIENT_ID",
                        help="this client holds its round-1 update (server must time it out)")
    parser.add_argument("--drop-after", type=float, default=2.5, dest="drop_after",
                        help="seconds after ROUND 1 starts before killing the client")
    parser.add_argument("--dp", action="store_true", help="enable client clipping + consensus noise")
    parser.add_argument("--timeout", type=float, default=420.0, help="global watchdog (seconds)")
    args = parser.parse_args(argv)

    if not (REPO_ROOT / "secret.key").exists():
        raise SystemExit("[smoke] missing secret.key; run tools/gen_key.py first")
    if not (REPO_ROOT / "data" / "processed" / "smoke" / "c1.npz").exists():
        raise SystemExit("[smoke] missing smoke files; run data/prepare.py first")

    if args.rounds:
        rounds = args.rounds
    elif args.drop:
        rounds = 12  # enough remaining rounds for the restart to land
    elif args.slow:
        rounds = 4
    else:
        rounds = 3
    clients = ["c1", "c2"]

    run_dir = REPO_ROOT / "runs" / f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = run_dir / "server_output"
    run_dir.mkdir(parents=True)

    server_cfg = {
        "network": {"host": "127.0.0.1", "port": args.port},
        "rounds": rounds,
        "expected_clients": len(clients),
        "expected_client_ids": clients,
        "min_clients": 1,
        "start_timeout_s": 120,
        "round_timeout_s": args.round_timeout,
        "training": {"local_epochs": 1, "kd_epochs": 2, "temperature": 2.0,
                     "batch_size": 64, "lr": 0.001, "optimizer": "adam"},
        "proxy": {"architecture": "cnn", "epochs": 2, "lr": 0.001},
        "dp": {"enabled": bool(args.dp), "clip_norm": 10.0, "noise_sigma": 1.0, "target_delta": 1e-5},
        "encryption": {"enabled": True, "key_file": "secret.key"},
        "paths": {"public_file": "data/processed/smoke/public.npz",
                  "test_file": "data/processed/smoke/test.npz",
                  "out_dir": str(out_dir)},
    }
    server_cfg_path = run_dir / "server_config.json"
    server_cfg_path.write_text(json.dumps(server_cfg, indent=2), encoding="utf-8")

    for client_id in clients:
        cfg = {
            "server_host": "127.0.0.1", "server_port": args.port, "client_id": client_id,
            "architecture": ARCHITECTURES.get(client_id, "cnn"),
            "shard_file": f"data/processed/smoke/{client_id}.npz",
            "public_file": "data/processed/smoke/public.npz",
            "encryption": {"enabled": True, "key_file": "secret.key"},
            "connect_timeout_s": 5, "retry_delay_s": 3,
            "local_metrics": True, "training_override": None, "log_file": None,
        }
        if args.slow == client_id:
            cfg["debug_delay"] = {"round": 1, "sleep_s": args.round_timeout + 10}
        if args.drop == client_id:
            # hold round 1 open long enough to be killed with certainty
            cfg["debug_delay"] = {"round": 1, "sleep_s": 60}
        (run_dir / f"client_{client_id}_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    server_log = out_dir / "server.log"
    procs = {}
    checks = []

    try:
        print(f"[smoke] run dir: {run_dir}")
        print(f"[smoke] rounds={rounds} port={args.port} round_timeout={args.round_timeout} "
              f"drop={args.drop} slow={args.slow} dp={args.dp}")

        procs["server"] = spawn(["-m", "server.server", "--config", str(server_cfg_path)],
                                run_dir / "server_console.log")
        if not wait_for(server_log, r"listening on", 60):
            raise SystemExit("[smoke] server did not start (check runs/.../server_console.log)")
        checks.append(("server started and listening", True))

        for client_id in clients:
            procs[client_id] = spawn(
                ["-m", "client.client", "--config", str(run_dir / f"client_{client_id}_config.json")],
                run_dir / f"client_{client_id}_console.log")
        if not wait_for(server_log, r"run starting with clients", 90):
            raise SystemExit("[smoke] clients never connected")
        checks.append(("clients connected", True))

        if not wait_for(server_log, r"=== ROUND 1", 90):
            raise SystemExit("[smoke] round 1 never started")
        checks.append(("round 1 started", True))

        involved = args.drop or args.slow
        if args.drop:
            time.sleep(args.drop_after)
            print(f"[smoke] killing client {args.drop} mid-round now")
            stop(procs[args.drop])
            print(f"[smoke] restarting {args.drop} immediately (it retries until the server accepts it)")
            procs[args.drop] = spawn(
                ["-m", "client.client", "--config", str(run_dir / f"client_{args.drop}_config.json")],
                run_dir / f"client_{args.drop}_restart_console.log")
            if not wait_for(server_log, rf"client {args.drop} .*(OFFLINE|did not answer)",
                            args.round_timeout + 120):
                raise SystemExit(f"[smoke] server never noticed {args.drop} going missing")
            checks.append((f"server noticed {args.drop} missing and kept the run alive", True))
            if not wait_for(server_log, rf"client {args.drop} reconnected", 150):
                raise SystemExit(f"[smoke] {args.drop} never reconnected")
            checks.append((f"{args.drop} reconnected by itself", True))
        elif args.slow:
            if not wait_for(server_log, rf"client {args.slow} did not answer within",
                            args.round_timeout + 120):
                raise SystemExit(f"[smoke] server never timed out {args.slow}")
            checks.append((f"server timed out {args.slow} and kept the run alive", True))

        if not wait_for(server_log, r"RUN FINISHED", args.timeout):
            raise SystemExit("[smoke] run did not finish in time")
        checks.append((f"all {rounds} rounds completed", True))

        # wrap-up checks
        server_code = wait_exit(procs["server"])
        checks.append((f"server exited cleanly (code {server_code})", server_code == 0))
        for client_id in clients:
            code = wait_exit(procs.get(client_id))
            checks.append((f"client {client_id} exited cleanly (code {code})", code == 0))

        metrics_path = out_dir / "metrics.csv"
        rows = []
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        checks.append((f"metrics.csv has {rounds + 1} rows (rounds 0..{rounds})", len(rows) == rounds + 1))
        final_acc = rows[-1]["test_acc"] if rows else ""
        try:
            acc_ok = float(final_acc) >= 0.10
        except (TypeError, ValueError):
            acc_ok = False
        if args.dp:
            checks.append((f"final test accuracy observed: {final_acc} (noise-path check: no accuracy requirement)", True))
            checks.append(("consensus noise applied", "privacy noise added" in server_log.read_text(encoding="utf-8", errors="replace")))
            client_path = out_dir / "client_metrics.csv"
            clipped = False
            if client_path.exists():
                with open(client_path, encoding="utf-8") as handle:
                    clipped = any(r.get("clipped") == "True" for r in csv.DictReader(handle))
            checks.append(("at least one update was clipped", clipped))
        else:
            checks.append((f"final test accuracy observed: {final_acc}", acc_ok))

        if involved:
            client_rows = []
            client_path = out_dir / "client_metrics.csv"
            if client_path.exists():
                with open(client_path, encoding="utf-8") as handle:
                    client_rows = list(csv.DictReader(handle))
            ok_rounds = {int(r["round"]) for r in client_rows
                         if r["client_id"] == involved and r["status"] == "ok"}
            checks.append((f"{involved}: at least one round missing/interrupted", len(ok_rounds) < rounds))
            checks.append((f"{involved}: back in the federation in the last round", rounds in ok_rounds))

        print()
        print("[smoke] ---- server log tail ----")
        try:
            tail = server_log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            for line in tail:
                print("    " + line)
        except OSError:
            pass
        print()
        print("[smoke] checklist:")
        for label, ok in checks:
            print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
        all_ok = all(ok for _, ok in checks)
        print(f"[smoke] RESULT: {'PASS' if all_ok else 'FAIL'}")
        sys.exit(0 if all_ok else 1)
    finally:
        stop(procs.get("server"))
        for client_id in clients:
            stop(procs.get(client_id))


if __name__ == "__main__":
    main()
