"""Federated learning server: rounds, FedAvg aggregation, optional DP, logs.

One round:
  1. broadcast the global weights + training config to all connected clients
  2. collect UPDATE messages until every expected client answered or
     round_timeout_s passes. A client that does not answer is logged and
     skipped for that round; the server never waits forever on anyone.
  3. aggregate: sample-weighted mean of the deltas (FedAvg); if DP is enabled,
     add calibrated Gaussian noise and log the running epsilon budget
  4. evaluate the new global model on the global test set, log the round

Run from the repo root:

    python -m server.server --config server/config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):  # allow `python server/server.py` as well
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data as data_mod
from common import privacy
from common import protocol
from common.model import make_model, param_count

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
log = logging.getLogger("server")


def _fmt(value, digits: int = 4):
    return "" if value is None else round(float(value), digits)


@dataclass(eq=False)
class ClientState:
    client_id: str
    conn: protocol.Connection
    n_train: int
    connected_at: float
    dead: bool = False
    misses: int = 0


class ClientReader(threading.Thread):
    """Reads one client socket and pushes ("msg"/"dead", state, payload) events."""

    def __init__(self, state: ClientState, events: "queue.Queue"):
        super().__init__(daemon=True, name=f"reader-{state.client_id}")
        self.state = state
        self.events = events

    def run(self) -> None:
        while True:
            try:
                message = self.state.conn.recv()
            except Exception as exc:
                self.events.put(("dead", self.state, exc))
                return
            self.events.put(("msg", self.state, message))


class Server:
    def __init__(self, config: dict):
        self.cfg = config
        net = config.get("network", {})
        self.host = str(net.get("host", "0.0.0.0"))
        self.port = int(net.get("port", 5000))
        self.rounds = int(config.get("rounds", 15))
        self.expected_clients = int(config.get("expected_clients", 3))
        self.expected_ids = [str(c) for c in config.get("expected_client_ids", [])]
        self.min_clients = int(config.get("min_clients", 1))
        self.start_timeout_s = float(config.get("start_timeout_s", 300))
        self.round_timeout_s = float(config.get("round_timeout_s", 30))
        self.training_cfg = dict(config.get("training", {}))

        paths = config.get("paths", {})
        self.test_file = str(paths.get("test_file", "data/processed/test.npz"))
        out_dir = paths.get("out_dir")
        self.out_dir = Path(out_dir) if out_dir else (
            Path(paths.get("out_root", "runs")) / datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        initial = make_model()
        self.param_names = list(initial.state_dict().keys())
        self.global_weights = {
            k: v.detach().cpu().numpy().astype(np.float32) for k, v in initial.state_dict().items()
        }
        self.model = make_model()
        self._load_global_into_model()

        dp = config.get("dp", {})
        self.dp_sigma = float(dp.get("noise_sigma", 0.0))
        self.dp_enabled = bool(dp.get("enabled", False)) and self.dp_sigma > 0
        self.dp_clip_norm = float(dp.get("clip_norm", 1.0))
        self.dp_target_delta = float(dp.get("target_delta", 1e-5))
        self.dp_rng = np.random.default_rng(dp.get("seed"))
        self.eps_round = (
            privacy.epsilon_gaussian_analytic(self.dp_sigma, self.dp_target_delta / max(self.rounds, 1))
            if self.dp_enabled else None
        )

        enc = config.get("encryption", {})
        self.key = protocol.load_key(enc.get("key_file", "secret.key")) if enc.get("enabled", True) else None

        self.clients: dict = {}
        self.events: "queue.Queue" = queue.Queue()
        self.metrics_rows: list = []
        self.metrics_columns = [
            "round", "wall_s", "clients_alive", "updates_ok",
            "test_loss", "test_acc", "agg_norm", "dp_sigma", "epsilon_spent",
        ]

    # ------------------------------------------------------------------ setup
    def setup_logging(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter(LOG_FORMAT)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        file_handler = logging.FileHandler(self.out_dir / "server.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.handlers[:] = [stream, file_handler]

    def _load_global_into_model(self) -> None:
        self.model.load_state_dict({k: torch.from_numpy(v) for k, v in self.global_weights.items()})

    def evaluate(self) -> tuple:
        return data_mod.evaluate(self.model, self.test_file)

    # --------------------------------------------------------------- clients
    def alive(self) -> list:
        return [s for s in self.clients.values() if not s.dead]

    def mark_dead(self, state: ClientState, reason: str) -> None:
        if state.dead:
            return
        state.dead = True
        state.conn.close()
        if self.clients.get(state.client_id) is state:
            log.warning("client %s marked OFFLINE (%s)", state.client_id, reason)
        else:
            log.info("stale connection of client %s closed (%s)", state.client_id, reason)

    def accept_one(self, listener: socket.socket) -> bool:
        try:
            sock, addr = listener.accept()
        except (socket.timeout, OSError):
            return False
        sock.settimeout(20.0)
        conn = protocol.from_socket(sock, self.key)
        try:
            hello = conn.recv()
        except Exception as exc:
            log.warning("dropping connection from %s: bad handshake (%s)", addr[0], exc)
            conn.close()
            return False
        if hello.get("type") != "HELLO" or not hello.get("client_id"):
            log.warning("dropping connection from %s: expected HELLO, got %r", addr[0], hello.get("type"))
            conn.close()
            return False
        cid = str(hello["client_id"])
        sock.settimeout(None)
        state = ClientState(cid, conn, int(hello.get("n_train", 0)), time.time())
        old = self.clients.get(cid)
        if old is not None:
            was = "was marked offline" if old.dead else "replacing previous connection"
            log.info("client %s reconnected (%s)", cid, was)
            old.dead = True
            old.conn.close()
        self.clients[cid] = state
        ClientReader(state, self.events).start()
        log.info("client %s connected from %s (n_train=%d)", cid, addr[0], state.n_train)
        try:
            state.conn.send({"type": "WELCOME", "proto": protocol.PROTO_VERSION,
                             "total_rounds": self.rounds})
        except Exception as exc:
            self.mark_dead(state, f"welcome send failed: {exc}")
            return False
        return True

    def wait_for_clients(self, listener: socket.socket) -> None:
        deadline = time.time() + self.start_timeout_s
        last_log = 0.0
        while True:
            self.accept_one(listener)
            connected = self.alive()
            if self.expected_ids:
                have = {s.client_id for s in connected}
                missing = [cid for cid in self.expected_ids if cid not in have]
                ready = not missing
            else:
                missing = []
                ready = len(connected) >= self.expected_clients
            if ready:
                break
            if time.time() > deadline:
                if len(connected) >= self.min_clients:
                    log.warning("start timeout reached; starting anyway with %d client(s): %s",
                                len(connected), sorted(s.client_id for s in connected))
                    break
                log.warning("start timeout reached but fewer than %d client(s) connected; still waiting ...",
                            self.min_clients)
                deadline = time.time() + 30
            elif time.time() - last_log > 15:
                last_log = time.time()
                target = self.expected_ids or f"{self.expected_clients} client(s)"
                log.info("waiting for clients: have %s, waiting for %s",
                         sorted(s.client_id for s in connected), missing or target)
        log.info("run starting with clients: %s", sorted(s.client_id for s in self.alive()))

    # ------------------------------------------------------------------ csvs
    def init_csvs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_file = open(self.out_dir / "metrics.csv", "w", newline="", encoding="utf-8")
        self._metrics_writer = csv.writer(self._metrics_file)
        self._metrics_writer.writerow(self.metrics_columns)
        self._client_file = open(self.out_dir / "client_metrics.csv", "w", newline="", encoding="utf-8")
        self._client_writer = csv.writer(self._client_file)
        self._client_writer.writerow(
            ["round", "client_id", "status", "n_train", "train_loss", "local_acc",
             "delta_norm", "clipped", "train_seconds"]
        )

    def record_round(self, r: int, wall_s: float, clients_alive: int, updates_ok: int,
                     loss: float, acc: float, agg_norm) -> None:
        eps_spent = round(self.eps_round * r, 3) if (self.dp_enabled and self.eps_round) else ""
        row = {
            "round": r,
            "wall_s": round(float(wall_s), 2),
            "clients_alive": clients_alive,
            "updates_ok": updates_ok,
            "test_loss": round(float(loss), 6),
            "test_acc": round(float(acc), 6),
            "agg_norm": "" if agg_norm is None else round(float(agg_norm), 4),
            "dp_sigma": self.dp_sigma if self.dp_enabled else "",
            "epsilon_spent": eps_spent,
        }
        self.metrics_rows.append(row)
        self._metrics_writer.writerow([row[c] for c in self.metrics_columns])
        self._metrics_file.flush()

    def write_client_row(self, r: int, state: ClientState, status: str, message: dict | None = None) -> None:
        aux = (message or {}).get("aux") or {}
        n_train = int((message or {}).get("n_train", state.n_train))
        self._client_writer.writerow([
            r, state.client_id, status, n_train,
            _fmt(aux.get("train_loss")), _fmt(aux.get("local_acc")),
            _fmt(aux.get("delta_norm")), aux.get("clipped", ""),
            _fmt(aux.get("train_seconds"), 2),
        ])
        self._client_file.flush()

    def write_run_config(self) -> None:
        meta = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "config": self.cfg,
            "model_params": param_count(make_model()),
            "param_names": self.param_names,
        }
        (self.out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def write_summary(self) -> None:
        try:
            rows = self.metrics_rows
            final = rows[-1] if rows else {}
            best = max((float(r["test_acc"]) for r in rows), default=0.0)
            summary = {
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "rounds_completed": int(final.get("round", 0)),
                "final_test_loss": final.get("test_loss"),
                "final_test_acc": final.get("test_acc"),
                "best_test_acc": best,
                "clients_seen": sorted(self.clients.keys()),
                "misses_per_client": {s.client_id: s.misses for s in self.clients.values()},
                "dp": (
                    {
                        "enabled": True,
                        "clip_norm": self.dp_clip_norm,
                        "noise_sigma": self.dp_sigma,
                        "target_delta": self.dp_target_delta,
                        "epsilon_total": round((self.eps_round or 0) * self.rounds, 3),
                        "note": "basic composition over planned rounds; conservative bound",
                    }
                    if self.dp_enabled else {"enabled": False}
                ),
            }
            (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception as exc:  # never let bookkeeping kill the process
            log.warning("could not write summary: %s", exc)

    # ---------------------------------------------------------------- rounds
    def round_config(self) -> dict:
        cfg = dict(self.training_cfg)
        if self.dp_enabled:
            cfg["clip_norm"] = self.dp_clip_norm
        return cfg

    def run_round(self, r: int, listener: socket.socket) -> None:
        last_warn = 0.0
        while not self.alive():
            if time.time() - last_warn > 15:
                log.warning("no connected clients; waiting for someone to come back ...")
                last_warn = time.time()
            time.sleep(1.0)
            self.accept_one(listener)
        log.info("=== ROUND %d/%d ===", r, self.rounds)
        started = time.time()
        message = {"type": "ROUND", "round": r, "weights": self.global_weights, "config": self.round_config()}
        expected = []
        for state in self.alive():
            try:
                state.conn.send(message)
                expected.append(state)
            except Exception as exc:
                self.mark_dead(state, f"send failed: {exc}")
        if not expected:
            log.warning("round %d: no reachable client; round skipped", r)
            loss, acc = self.evaluate()
            self.record_round(r, time.time() - started, len(self.alive()), 0, loss, acc, None)
            return
        updates = self.collect_updates(r, expected, listener)
        agg, agg_norm = self.aggregate(updates)
        if agg is None:
            log.warning("round %d: no valid updates; global model unchanged", r)
            loss, acc = self.evaluate()
            self.record_round(r, time.time() - started, len(self.alive()), 0, loss, acc, None)
            return
        if self.dp_enabled:
            counts = [int(m.get("n_train", 0)) for _, m in updates if int(m.get("n_train", 0)) > 0]
            sens = privacy.mean_sensitivity(self.dp_clip_norm, counts)
            agg = privacy.add_gaussian_noise(agg, self.dp_sigma, sens, self.dp_rng)
            log.info("DP noise added: sigma=%.2f sensitivity=%.4f", self.dp_sigma, sens)
        for k in self.global_weights:
            self.global_weights[k] = (self.global_weights[k] + agg[k]).astype(np.float32)
        self._load_global_into_model()
        loss, acc = self.evaluate()
        elapsed = time.time() - started
        self.record_round(r, elapsed, len(self.alive()), len(updates), loss, acc, agg_norm)
        log.info("round %d done in %.1f s | %d update(s) | test_loss=%.4f test_acc=%.4f",
                 r, elapsed, len(updates), loss, acc)

    def collect_updates(self, r: int, expected: list, listener: socket.socket) -> list:
        pending = {s: True for s in expected}
        updates = []
        deadline = time.time() + self.round_timeout_s
        while pending:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                kind, state, payload = self.events.get(timeout=min(0.25, remaining))
            except queue.Empty:
                self.accept_one(listener)  # register reconnecting clients promptly
                continue
            if kind == "dead":
                if state in pending:
                    del pending[state]
                    self.mark_dead(state, "connection lost while waiting for its update")
                    self.write_client_row(r, state, "dead")
                else:
                    state.dead = True
                continue
            message = payload
            if state not in pending:
                continue  # late update from a previous round, or duplicate: ignored
            if message.get("type") != "UPDATE" or int(message.get("round", -1)) != r:
                log.warning("client %s sent %r for round %s while round %d is open; ignored",
                            state.client_id, message.get("type"), message.get("round"), r)
                continue
            delta = message.get("delta")
            if (not isinstance(delta, dict)
                    or set(delta.keys()) != set(self.global_weights)
                    or any(np.asarray(delta[k]).shape != self.global_weights[k].shape for k in delta)):
                log.error("client %s sent a malformed delta; update dropped", state.client_id)
                del pending[state]
                self.write_client_row(r, state, "bad-delta")
                continue
            del pending[state]
            updates.append((state, message))
            self.write_client_row(r, state, "ok", message)
        for state in pending:
            state.misses += 1
            log.info("client %s did not answer within %.0f s; skipped for round %d (total misses: %d)",
                     state.client_id, self.round_timeout_s, r, state.misses)
            self.write_client_row(r, state, "miss")
        return updates

    def aggregate(self, updates: list) -> tuple:
        usable = [(s, m) for s, m in updates if int(m.get("n_train", 0)) > 0]
        if not usable:
            return None, 0.0
        total_n = sum(int(m["n_train"]) for _, m in usable)
        agg = {k: np.zeros_like(v) for k, v in self.global_weights.items()}
        for _, m in usable:
            weight = int(m["n_train"]) / total_n
            for k, arr in m["delta"].items():
                agg[k] = (agg[k] + (weight * np.asarray(arr, dtype=np.float32))).astype(np.float32)
        return agg, privacy.l2_norm(agg)

    def finish(self) -> None:
        torch.save(self.model.state_dict(), self.out_dir / "model_final.pt")
        log.info("saved final model: %s", self.out_dir / "model_final.pt")
        for state in self.alive():
            try:
                state.conn.send({"type": "FINISHED"})
            except Exception:
                pass
        time.sleep(0.3)

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        self.setup_logging()
        log.info("=" * 70)
        log.info("FedAvg server starting | rounds=%d | expected=%s | round_timeout=%.0f s",
                 self.rounds, self.expected_ids or f"{self.expected_clients} client(s)", self.round_timeout_s)
        log.info("training config: %s", json.dumps(self.round_config()))
        if self.dp_enabled:
            eps_total = privacy.total_epsilon(self.dp_sigma, self.rounds, self.dp_target_delta)
            log.info("DP: clip_norm=%.2f sigma=%.2f target_delta=%.1e -> epsilon ~= %.2f over %d rounds (basic composition)",
                     self.dp_clip_norm, self.dp_sigma, self.dp_target_delta, eps_total, self.rounds)
        else:
            log.info("DP: disabled (plain FedAvg)")
        log.info("output dir: %s", self.out_dir)

        listener = protocol.make_listener(self.host, self.port)
        listener.settimeout(0.5)
        log.info("listening on %s:%d", self.host, self.port)
        try:
            self.wait_for_clients(listener)
            self.write_run_config()
            self.init_csvs()
            loss, acc = self.evaluate()
            self.record_round(0, 0.0, len(self.alive()), 0, loss, acc, None)
            log.info("round 0 (initial random model): test_loss=%.4f test_acc=%.4f", loss, acc)
            for r in range(1, self.rounds + 1):
                self.run_round(r, listener)
            self.finish()
            final = self.metrics_rows[-1]
            log.info("RUN FINISHED | final test_acc=%.4f over %s round(s)",
                     final["test_acc"], final["round"])
            if self.dp_enabled:
                log.info("privacy spent: epsilon ~= %.2f at delta=%.1e over %d rounds (basic composition)",
                         (self.eps_round or 0) * self.rounds, self.dp_target_delta, self.rounds)
        except KeyboardInterrupt:
            log.warning("interrupted by user")
        finally:
            for state in self.clients.values():
                state.conn.close()
            listener.close()
            self.write_summary()
            for attr in ("_metrics_file", "_client_file"):
                handle = getattr(self, attr, None)
                if handle is not None:
                    handle.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Federated learning server")
    parser.add_argument("--config", default="server/config.json")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--round-timeout", type=float, default=None, dest="round_timeout")
    parser.add_argument("--dp", action="store_true", help="force DP on (clip + Gaussian noise)")
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--clip", type=float, default=None)
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.port is not None:
        config.setdefault("network", {})["port"] = args.port
    if args.rounds is not None:
        config["rounds"] = args.rounds
    if args.round_timeout is not None:
        config["round_timeout_s"] = args.round_timeout
    if args.dp:
        config.setdefault("dp", {})["enabled"] = True
    if args.sigma is not None:
        config.setdefault("dp", {})["noise_sigma"] = args.sigma
    if args.clip is not None:
        config.setdefault("dp", {})["clip_norm"] = args.clip
    Server(config).run()


if __name__ == "__main__":
    main()
