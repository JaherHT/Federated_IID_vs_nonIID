"""Federated learning server: logits-exchange rounds, logs, CSV metrics.

The team's strategy is federated distillation over a public set instead of
FedAvg: clients never send model weights, so every client can run its own
architecture. Each round:

  1. broadcast ROUND (current consensus probabilities on the public set + the
     training config) to every connected client
  2. collect UPDATE messages (each client's probabilities on the public set)
     until every expected client answered or round_timeout_s passes. A client
     that does not answer is logged and skipped; the server never waits
     indefinitely on anyone.
  3. aggregate: sample-weighted mean of the probabilities -> new consensus
  4. train the server-side proxy model on the public set toward the consensus,
     then evaluate it on the held-out test set and log the round

The connection is strict: the server accepts a HELLO handshake once, then
UPDATE messages for the open round. Anything else closes the connection.
Payloads are plain data (JSON + float32 arrays); nothing received over the
network is ever executed, imported, or written to disk.

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
from common.model import build_model, param_count

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
log = logging.getLogger("server")

ALLOWED_AUX_KEYS = {"local_loss", "local_acc", "distill_loss", "probs_norm",
                    "clipped", "train_seconds"}


def _fmt(value, digits: int = 4):
    return "" if value is None else round(float(value), digits)


@dataclass(eq=False)
class ClientState:
    client_id: str
    conn: protocol.Connection
    n_train: int
    arch: str
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
        self.round_timeout_s = float(config.get("round_timeout_s", 90))
        self.training_cfg = dict(config.get("training", {}))
        self.strategy = str(config.get("strategy", "logits")).lower()
        if self.strategy not in ("logits", "fedavg"):
            raise ValueError(f"unknown strategy {self.strategy!r}; use 'logits' or 'fedavg'")
        model_cfg = config.get("model", {})
        self.model_arch = str(model_cfg.get("architecture", "cnn"))
        self.distill_epochs = int(model_cfg.get("distill_epochs", 2))
        self.distill_lr = float(model_cfg.get("distill_lr", 0.001))

        paths = config.get("paths", {})
        self.public_file = str(paths.get("public_file", "data/processed/public.npz"))
        self.test_file = str(paths.get("test_file", "data/processed/test.npz"))
        out_dir = paths.get("out_dir")
        self.out_dir = Path(out_dir) if out_dir else (
            Path(paths.get("out_root", "runs")) / datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        with np.load(self.public_file) as payload:
            self.public_X_t = data_mod.normalize_images(payload["X"].copy())
        self.n_public = int(self.public_X_t.shape[0])
        self.num_classes = 10

        self.model = build_model(self.model_arch)
        self.model_params = param_count(self.model)
        self.consensus = None  # np.ndarray (n_public, 10) float32 after round 1

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

    def evaluate(self) -> tuple:
        return data_mod.evaluate(self.model, self.test_file)

    def _weights_np(self) -> dict:
        return {name: tensor.detach().cpu().numpy().astype(np.float32)
                for name, tensor in self.model.state_dict().items()}

    def _load_weights(self, arrays: dict) -> None:
        state = {name: torch.from_numpy(np.asarray(arr, dtype=np.float32))
                 for name, arr in arrays.items()}
        self.model.load_state_dict(state)

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

    def _hello_problem(self, hello: dict) -> str:
        if hello.get("type") != "HELLO":
            return f"expected HELLO first, got {hello.get('type')!r}"
        if int(hello.get("proto", -1)) != protocol.PROTO_VERSION:
            return f"protocol version mismatch (client sent {hello.get('proto')!r})"
        cid = hello.get("client_id")
        if not isinstance(cid, str) or not (0 < len(cid) <= 64):
            return f"bad client_id {cid!r}"
        if self.expected_ids and cid not in self.expected_ids:
            return f"unknown client_id {cid!r} (expected {self.expected_ids})"
        if hello.get("ready") is not True:
            return "client not ready (model/data must be loaded before joining)"
        n_train = hello.get("n_train")
        if not isinstance(n_train, int) or not (0 < n_train <= 10_000_000):
            return f"bad n_train {n_train!r}"
        arch = hello.get("arch")
        if not isinstance(arch, str) or not (0 < len(arch) <= 64):
            return f"bad arch descriptor {arch!r}"
        return ""

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
        problem = self._hello_problem(hello)
        if problem:
            log.warning("refused connection from %s: %s", addr[0], problem)
            conn.close()
            return False
        cid = str(hello["client_id"])
        sock.settimeout(None)
        state = ClientState(cid, conn, int(hello["n_train"]), str(hello["arch"]), time.time())
        old = self.clients.get(cid)
        if old is not None:
            was = "was marked offline" if old.dead else "replacing previous connection"
            log.info("client %s reconnected (%s)", cid, was)
            old.dead = True
            old.conn.close()
        self.clients[cid] = state
        ClientReader(state, self.events).start()
        log.info("client %s connected from %s (n_train=%d, arch=%s)", cid, addr[0], state.n_train, state.arch)
        if self.strategy == "fedavg" and state.arch != self.model_arch:
            log.warning("client %s runs arch=%s but FedAvg needs the shared model '%s' on every client",
                        cid, state.arch, self.model_arch)
        try:
            state.conn.send({"type": "WELCOME", "proto": protocol.PROTO_VERSION,
                             "total_rounds": self.rounds, "public_size": self.n_public,
                             "num_classes": self.num_classes, "strategy": self.strategy,
                             "shared_model": self.model_arch})
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
            ["round", "client_id", "status", "n_train", "local_loss", "local_acc",
             "distill_loss", "update_norm", "clipped", "train_seconds"]
        )
    # --------------------------------------------------------------- records
    def record_round(self, r: int, wall_s: float, clients_alive: int, updates_ok: int,
                     loss: float, acc: float, consensus_norm) -> None:
        eps_spent = round(self.eps_round * r, 3) if (self.dp_enabled and self.eps_round) else ""
        row = {
            "round": r,
            "wall_s": round(float(wall_s), 2),
            "clients_alive": clients_alive,
            "updates_ok": updates_ok,
            "test_loss": round(float(loss), 6),
            "test_acc": round(float(acc), 6),
            "agg_norm": "" if consensus_norm is None else round(float(consensus_norm), 4),
            "dp_sigma": self.dp_sigma if self.dp_enabled else "",
            "epsilon_spent": eps_spent,
        }
        self.metrics_rows.append(row)
        self._metrics_writer.writerow([row[c] for c in self.metrics_columns])
        self._metrics_file.flush()

    def write_client_row(self, r: int, state: ClientState, status: str, message: dict | None = None) -> None:
        aux = (message or {}).get("aux") or {}
        n_train = int((message or {}).get("n_train", state.n_train))
        update_norm = aux.get("probs_norm", aux.get("delta_norm"))
        self._client_writer.writerow([
            r, state.client_id, status, n_train,
            _fmt(aux.get("local_loss")), _fmt(aux.get("local_acc")),
            _fmt(aux.get("distill_loss")), _fmt(update_norm),
            aux.get("clipped", ""), _fmt(aux.get("train_seconds"), 2),
        ])
        self._client_file.flush()

    def write_run_config(self) -> None:
        meta = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "config": self.cfg,
            "strategy": self.strategy,
            "model_architecture": self.model_arch,
            "model_params": self.model_params,
            "public_samples": self.n_public,
        }
        (self.out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def write_summary(self) -> None:
        try:
            rows = self.metrics_rows
            final = rows[-1] if rows else {}
            best = max((float(r["test_acc"]) for r in rows), default=0.0)
            summary = {
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "strategy": self.strategy,
                "rounds_completed": int(final.get("round", 0)),
                "final_test_loss": final.get("test_loss"),
                "final_test_acc": final.get("test_acc"),
                "best_test_acc": best,
                "clients_seen": {cid: s.arch for cid, s in sorted(self.clients.items())},
                "misses_per_client": {s.client_id: s.misses for s in self.clients.values()},
            }
            if self.dp_enabled:
                summary["privacy"] = {
                    "clip_norm": self.dp_clip_norm,
                    "noise_sigma": self.dp_sigma,
                    "target_delta": self.dp_target_delta,
                    "epsilon_total": round((self.eps_round or 0) * self.rounds, 3),
                    "note": "basic composition over planned rounds; conservative bound",
                }
            (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception as exc:  # never let bookkeeping kill the process
            log.warning("could not write summary: %s", exc)

    # ---------------------------------------------------------------- rounds
    def round_config(self) -> dict:
        cfg = {
            "local_epochs": int(self.training_cfg.get("local_epochs", 1)),
            "kd_epochs": int(self.training_cfg.get("kd_epochs", 2)),
            "temperature": float(self.training_cfg.get("temperature", 2.0)),
            "batch_size": int(self.training_cfg.get("batch_size", 64)),
            "lr": float(self.training_cfg.get("lr", 0.001)),
            "optimizer": str(self.training_cfg.get("optimizer", "adam")),
        }
        if self.dp_enabled:
            cfg["clip_norm"] = self.dp_clip_norm
        return cfg

    @staticmethod
    def _renormalize(probs: np.ndarray) -> np.ndarray:
        probs = np.clip(np.asarray(probs, dtype=np.float32), 0.0, None)
        sums = probs.sum(axis=1, keepdims=True)
        bad = sums[:, 0] <= 0
        if bool(bad.any()):
            probs[bad] = 1.0 / probs.shape[1]
            sums = probs.sum(axis=1, keepdims=True)
        return (probs / sums).astype(np.float32)

    def _train_proxy(self) -> None:
        """Train the server-side proxy model on the public set toward the consensus."""
        if self.consensus is None:
            return
        cfg = self.round_config()
        temperature = float(cfg["temperature"])
        batch_size = int(cfg["batch_size"])
        targets = torch.from_numpy(self.consensus)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.distill_lr)
        n = self.n_public
        self.model.train()
        started = time.time()
        for _ in range(self.distill_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                logits = self.model(self.public_X_t[idx])
                logp = torch.log_softmax(logits / temperature, dim=1)
                loss = torch.nn.functional.kl_div(logp, targets[idx], reduction="batchmean") * (temperature * temperature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        log.info("server model distilled on %d public samples x %d epoch(s) in %.1f s",
                 n, self.distill_epochs, time.time() - started)

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
        if self.strategy == "fedavg":
            message = {"type": "ROUND", "round": r, "weights": self._weights_np(),
                       "config": self.round_config()}
        else:
            message = {"type": "ROUND", "round": r, "targets": self.consensus,
                       "config": self.round_config()}
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
        if self.strategy == "fedavg":
            self.apply_fedavg(updates, r, started)
            return
        consensus_new, cons_norm = self.aggregate(updates)
        if consensus_new is None:
            log.warning("round %d: no valid updates; consensus unchanged", r)
            loss, acc = self.evaluate()
            self.record_round(r, time.time() - started, len(self.alive()), 0, loss, acc, None)
            return
        if self.dp_enabled:
            counts = [int(m.get("n_train", 0)) for _, m in updates if int(m.get("n_train", 0)) > 0]
            sens = privacy.mean_sensitivity(self.dp_clip_norm, counts)
            consensus_new = privacy.add_gaussian_noise_array(consensus_new, self.dp_sigma, sens, self.dp_rng)
            log.info("privacy noise added: sigma=%.2f sensitivity=%.4f", self.dp_sigma, sens)
        self.consensus = self._renormalize(consensus_new)
        self._train_proxy()
        loss, acc = self.evaluate()
        elapsed = time.time() - started
        self.record_round(r, elapsed, len(self.alive()), len(updates), loss, acc, cons_norm)
        log.info("round %d done in %.1f s | %d update(s) | test_loss=%.4f test_acc=%.4f",
                 r, elapsed, len(updates), loss, acc)

    def _update_problem(self, message: dict, r: int, state: ClientState) -> str:
        """Reason string if this message must drop the connection, '' if fine.

        Late updates (for older rounds) never reach this method; the collect
        loop ignores them before calling here.
        """
        if message.get("type") != "UPDATE":
            return f"expected UPDATE, got {message.get('type')!r}"
        if not isinstance(message.get("round"), int) or message.get("round") != r:
            return f"invalid round field {message.get('round')!r}"
        if message.get("client_id") != state.client_id:
            return "client_id does not match the handshake"
        n_train = message.get("n_train")
        if not isinstance(n_train, int) or not (0 < n_train <= 10_000_000):
            return f"bad n_train {n_train!r}"
        if self.strategy == "fedavg":
            return self._delta_problem(message.get("delta"))
        return self._probs_problem(message.get("probs"))

    def _probs_problem(self, probs) -> str:
        if not isinstance(probs, np.ndarray):
            return "missing probability array"
        if probs.dtype != np.float32:
            return f"probabilities must be float32 (got {probs.dtype})"
        if probs.ndim != 2 or probs.shape[1] != self.num_classes or probs.shape[0] != self.n_public:
            return (f"probability shape mismatch (got {tuple(probs.shape)}, "
                    f"expected ({self.n_public}, {self.num_classes}))")
        if not bool(np.isfinite(probs).all()):
            return "non-finite values"
        if bool((probs < -1e-6).any()) or bool((probs > 1.0 + 1e-6).any()):
            return "values outside [0, 1]"
        sums = probs.sum(axis=1)
        if bool((sums < 0.9).any()) or bool((sums > 1.1).any()):
            return "rows do not sum to ~1"
        return ""

    def _delta_problem(self, delta) -> str:
        if not isinstance(delta, dict):
            return "missing delta dict"
        expected = self.model.state_dict()
        if set(delta.keys()) != set(expected.keys()):
            return "delta keys do not match the shared model"
        for name, tensor in expected.items():
            arr = delta[name]
            if not isinstance(arr, np.ndarray) or arr.dtype != np.float32:
                return f"delta[{name!r}] must be a float32 array"
            if tuple(arr.shape) != tuple(tensor.shape):
                return f"delta[{name!r}] shape mismatch"
            if not bool(np.isfinite(arr).all()):
                return f"delta[{name!r}] has non-finite values"
        largest = max((float(np.abs(arr).max()) for arr in delta.values()), default=0.0)
        if largest > 1000.0:
            return "delta values out of range"
        return ""

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
                continue  # duplicate, or from a round that is already closed: ignored
            round_field = message.get("round")
            if message.get("type") == "UPDATE" and isinstance(round_field, int) and round_field != r:
                log.warning("client %s sent a late update for round %s (round %d is open); ignoring it",
                            state.client_id, round_field, r)
                continue
            problem = self._update_problem(message, r, state)
            if problem:
                log.error("client %s sent a message that breaks the protocol (%s); connection dropped",
                          state.client_id, problem)
                del pending[state]
                self.write_client_row(r, state, "bad-update")
                self.mark_dead(state, f"protocol violation: {problem}")
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
        agg = np.zeros_like(np.asarray(usable[0][1]["probs"]), dtype=np.float32)
        for _, m in usable:
            weight = int(m["n_train"]) / total_n
            agg = (agg + (weight * np.asarray(m["probs"], dtype=np.float32))).astype(np.float32)
        return agg, privacy.l2_norm_array(agg)

    def apply_fedavg(self, updates: list, r: int, started: float) -> None:
        """Weighted mean of client deltas, applied to the server-side model."""
        usable = [(s, m) for s, m in updates if int(m.get("n_train", 0)) > 0]
        if not usable:
            log.warning("round %d: no valid updates; global model unchanged", r)
            loss, acc = self.evaluate()
            self.record_round(r, time.time() - started, len(self.alive()), 0, loss, acc, None)
            return
        total_n = sum(int(m["n_train"]) for _, m in usable)
        base = self._weights_np()
        mean_delta = {name: np.zeros_like(arr) for name, arr in base.items()}
        for _, m in usable:
            weight = int(m["n_train"]) / total_n
            for name, arr in m["delta"].items():
                mean_delta[name] = (mean_delta[name]
                                    + weight * np.asarray(arr, dtype=np.float32)).astype(np.float32)
        if self.dp_enabled:
            counts = [int(m.get("n_train", 0)) for _, m in usable]
            sens = privacy.mean_sensitivity(self.dp_clip_norm, counts)
            mean_delta = privacy.add_gaussian_noise_dict(mean_delta, self.dp_sigma, sens, self.dp_rng)
            log.info("privacy noise added: sigma=%.2f sensitivity=%.4f", self.dp_sigma, sens)
        norm = privacy.l2_norm_dict(mean_delta)
        self._load_weights({name: (base[name] + mean_delta[name]).astype(np.float32) for name in base})
        loss, acc = self.evaluate()
        elapsed = time.time() - started
        self.record_round(r, elapsed, len(self.alive()), len(updates), loss, acc, norm)
        log.info("round %d done in %.1f s | %d update(s) | test_loss=%.4f test_acc=%.4f",
                 r, elapsed, len(updates), loss, acc)

    def finish(self) -> None:
        torch.save(self.model.state_dict(), self.out_dir / "model_final.pt")
        log.info("saved final model: %s", self.out_dir / "model_final.pt")
        if self.consensus is not None:
            np.save(self.out_dir / "consensus_final.npy", self.consensus)
            log.info("saved final consensus: %s", self.out_dir / "consensus_final.npy")
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
        log.info("federated server starting | strategy=%s | rounds=%d | expected=%s | round_timeout=%.0f s",
                 self.strategy, self.rounds,
                 self.expected_ids or f"{self.expected_clients} client(s)", self.round_timeout_s)
        log.info("training config: %s", json.dumps(self.round_config()))
        log.info("server model: %s (%d parameters)", self.model_arch, self.model_params)
        if self.dp_enabled:
            eps_total = privacy.total_epsilon(self.dp_sigma, self.rounds, self.dp_target_delta)
            log.info("privacy: clip_norm=%.2f sigma=%.2f target_delta=%.1e -> epsilon ~= %.2f over %d rounds (basic composition)",
                     self.dp_clip_norm, self.dp_sigma, self.dp_target_delta, eps_total, self.rounds)
        log.info("public set: %d samples (%s) | test set: %s", self.n_public, self.public_file, self.test_file)
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
            log.info("round 0 (untrained proxy model): test_loss=%.4f test_acc=%.4f", loss, acc)
            for r in range(1, self.rounds + 1):
                self.run_round(r, listener)
            self.finish()
            final = self.metrics_rows[-1]
            log.info("RUN FINISHED | final test_acc=%.4f over %s round(s)", final["test_acc"], final["round"])
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
    parser = argparse.ArgumentParser(description="Federated learning server (logits exchange)")
    parser.add_argument("--config", default="server/config.json")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--round-timeout", type=float, default=None, dest="round_timeout")
    parser.add_argument("--dp", action="store_true", help="enable client clipping + consensus noise")
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
