"""Federated learning client (reference implementation).

You define your own model here: pick one of the presets in common/model.py
("cnn", "mlp", "cnn_deep") or add your own. The federation exchanges only
probabilities on a shared public set, so your architecture is yours alone.

One round:
  ROUND message arrives
    -> train locally on your private shard (cross-entropy)
    -> distill from the consensus probabilities on the public set
       (skipped in round 1, when there is no consensus yet)
    -> compute your model's probabilities on the public set
    -> send UPDATE {round, client_id, n_train, probs, aux}

The client loads model and data before connecting, and it retries by itself
after network drops; restarting is never required.

Run from the repo root:

    python -m client.client --config client/config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):  # allow `python client/client.py` as well
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data as data_mod
from common import privacy
from common import protocol
from common.model import build_model, param_count

log = logging.getLogger("client")


def _f(value):
    return None if value is None else float(value)


class Client:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.host = str(cfg["server_host"])
        self.port = int(cfg["server_port"])
        self.client_id = str(cfg["client_id"])
        self.arch = str(cfg.get("architecture", "cnn"))
        self.shard_file = str(cfg.get("shard_file", f"data/processed/iid/{self.client_id}.npz"))
        self.public_file = str(cfg.get("public_file", "data/processed/public.npz"))
        self.connect_timeout_s = float(cfg.get("connect_timeout_s", 10))
        self.retry_delay_s = float(cfg.get("retry_delay_s", 3))
        self.debug_delay = cfg.get("debug_delay") or {}
        self.training_override = cfg.get("training_override") or {}
        log_file = cfg.get("log_file")
        self.history: list = []
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = Path(cfg.get("out_dir") or Path("client_out") / f"{self.client_id}_{stamp}")

        enc = cfg.get("encryption", {})
        self.key = protocol.load_key(enc.get("key_file", "secret.key")) if enc.get("enabled", True) else None

        # Load everything BEFORE connecting: the server only accepts clients
        # whose model and data are ready.
        shard = data_mod.Shard(self.shard_file)
        self.n_train = len(shard)
        self.shard_path = self.shard_file
        self.public_X = data_mod.load_inputs(self.public_file)
        self.model = build_model(self.arch)

        logger_root = logging.getLogger()
        logger_root.setLevel(logging.INFO)
        handlers = [logging.StreamHandler(sys.stdout)]
        if log_file:
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        fmt = logging.Formatter(f"%(asctime)s | %(levelname)-7s | [{self.client_id}] %(message)s")
        for handler in handlers:
            handler.setFormatter(fmt)
        logger_root.handlers[:] = handlers

        log.info("config: server=%s:%d arch=%s | shard=%s (%d samples) | public=%s (%d samples) | %d params",
                 self.host, self.port, self.arch, self.shard_file, self.n_train,
                 self.public_file, int(self.public_X.shape[0]), param_count(self.model))

    # -------------------------------------------------------------- training
    def _training_plan(self, round_cfg: dict) -> dict:
        plan = {
            "local_epochs": 1,
            "kd_epochs": 2,
            "temperature": 2.0,
            "batch_size": 64,
            "lr": 0.001,
            "optimizer": "adam",
        }
        allowed = set(plan) | {"clip_norm"}
        plan.update({k: v for k, v in (round_cfg or {}).items() if k in allowed})
        plan.update({k: v for k, v in self.training_override.items() if k in allowed})
        return plan

    def _optimizer(self, plan: dict):
        if str(plan.get("optimizer", "adam")) == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=float(plan["lr"]))
        return torch.optim.Adam(self.model.parameters(), lr=float(plan["lr"]))

    def train_local(self, plan: dict) -> float:
        self.model.train()
        loader = data_mod.make_loader(self.shard_path, batch_size=int(plan["batch_size"]), shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self._optimizer(plan)
        last = 0.0
        for _ in range(int(plan["local_epochs"])):
            total, batches = 0.0, 0
            for x, y in loader:
                loss = criterion(self.model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            last = total / max(batches, 1)
        return last

    def distill(self, targets, plan: dict) -> float:
        temperature = float(plan["temperature"])
        target_tensor = torch.from_numpy(np.asarray(targets, dtype=np.float32))
        self.model.train()
        optimizer = self._optimizer(plan)
        n = int(self.public_X.shape[0])
        batch_size = int(plan["batch_size"])
        last = 0.0
        for _ in range(int(plan["kd_epochs"])):
            perm = torch.randperm(n)
            total, batches = 0.0, 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                logits = self.model(self.public_X[idx])
                logp = F.log_softmax(logits / temperature, dim=1)
                loss = F.kl_div(logp, target_tensor[idx], reduction="batchmean") * (temperature * temperature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            last = total / max(batches, 1)
        return last

    @torch.no_grad()
    def public_probs(self, temperature: float) -> np.ndarray:
        """Softmax probabilities of the current model over the public set."""
        self.model.eval()
        chunks = []
        n = int(self.public_X.shape[0])
        for i in range(0, n, 512):
            logits = self.model(self.public_X[i:i + 512])
            probs = torch.softmax(logits / temperature, dim=1)
            chunks.append(probs.cpu().numpy().astype(np.float32))
        self.model.train()
        if not chunks:
            return np.zeros((0, 10), dtype=np.float32)
        return np.concatenate(chunks, axis=0)
    # ----------------------------------------------------------------- round
    def handle_round(self, message: dict, conn: protocol.Connection) -> None:
        if "weights" in message:
            self.handle_round_weights(message, conn)
        elif "targets" in message:
            self.handle_round_probs(message, conn)
        else:
            raise protocol.ProtocolError("ROUND carries neither weights nor targets")

    def handle_round_weights(self, message: dict, conn: protocol.Connection) -> None:
        """FedAvg round: load the shared weights, train, send the delta."""
        r = int(message["round"])
        plan = self._training_plan(message.get("config"))
        started = time.time()

        received = {name: np.asarray(arr, dtype=np.float32)
                    for name, arr in message["weights"].items()}
        try:
            self.model.load_state_dict({name: torch.from_numpy(arr) for name, arr in received.items()})
        except Exception as exc:
            raise protocol.ProtocolError(
                f"received weights do not fit architecture '{self.arch}' ({exc}); "
                "FedAvg runs need the shared model on every client") from exc

        local_loss = None
        if int(plan["local_epochs"]) > 0:
            local_loss = self.train_local(plan)

        trained = {name: tensor.detach().cpu().numpy().astype(np.float32)
                   for name, tensor in self.model.state_dict().items()}
        delta = {name: (trained[name] - received[name]).astype(np.float32) for name in received}
        delta_norm = privacy.l2_norm_dict(delta)
        clipped = False
        clip_norm = plan.get("clip_norm")
        if clip_norm is not None:
            delta, clipped, delta_norm = privacy.clip_dict(delta, float(clip_norm))

        train_seconds = time.time() - started
        if self.debug_delay.get("round") == r:
            sleep_s = float(self.debug_delay.get("sleep_s", 0.0))
            if sleep_s > 0:
                log.warning("debug_delay active: holding the round-%d update for %.0f s", r, sleep_s)
                time.sleep(sleep_s)

        if not all(bool(np.isfinite(arr).all()) for arr in delta.values()):
            log.error("round %d: model produced a non-finite delta; skipping this update", r)
            return

        local_acc = None
        if self.cfg.get("local_metrics", True):
            _, local_acc = data_mod.evaluate(self.model, self.shard_path)

        aux = {
            "local_loss": _f(local_loss),
            "local_acc": _f(local_acc),
            "delta_norm": _f(delta_norm),
            "clipped": bool(clipped),
            "train_seconds": _f(train_seconds),
        }
        aux = {k: v for k, v in aux.items() if v is not None}
        self.history.append({"round": r, "n_train": int(self.n_train), **aux})

        conn.send({
            "type": "UPDATE",
            "round": r,
            "client_id": self.client_id,
            "n_train": int(self.n_train),
            "delta": delta,
            "aux": aux,
        })
        log.info("round %d: delta sent (local_loss=%s delta_norm=%.3f clipped=%s | %.1f s)",
                 r, aux.get("local_loss"), aux.get("delta_norm", 0.0), clipped, train_seconds)

    def handle_round_probs(self, message: dict, conn: protocol.Connection) -> None:
        r = int(message["round"])
        plan = self._training_plan(message.get("config"))
        started = time.time()

        local_loss = None
        if int(plan["local_epochs"]) > 0:
            local_loss = self.train_local(plan)

        targets = message.get("targets")
        distill_loss = None
        if targets is not None and int(plan["kd_epochs"]) > 0:
            distill_loss = self.distill(targets, plan)

        probs = self.public_probs(float(plan["temperature"]))
        # Clipping (when the round asks for it) applies to the deviation from
        # the uniform distribution: the deviation bounds the update's norm,
        # and adding the uniform layer back keeps every row a probability row
        # (elements in [0, 1], sums to 1) as the protocol requires.
        uniform = 1.0 / probs.shape[1]
        deviation = (probs - uniform).astype(np.float32)
        probs_norm = privacy.l2_norm_array(deviation)
        clipped = False
        clip_norm = plan.get("clip_norm")
        if clip_norm is not None:
            deviation, clipped, probs_norm = privacy.clip_array(deviation, float(clip_norm))
            probs = (deviation + uniform).astype(np.float32)

        train_seconds = time.time() - started
        if self.debug_delay.get("round") == r:
            sleep_s = float(self.debug_delay.get("sleep_s", 0.0))
            if sleep_s > 0:
                log.warning("debug_delay active: holding the round-%d update for %.0f s", r, sleep_s)
                time.sleep(sleep_s)

        if not bool(np.isfinite(probs).all()):
            log.error("round %d: model produced non-finite probabilities; skipping this update", r)
            return

        local_acc = None
        if self.cfg.get("local_metrics", True):
            _, local_acc = data_mod.evaluate(self.model, self.shard_path)

        aux = {
            "local_loss": _f(local_loss),
            "local_acc": _f(local_acc),
            "distill_loss": _f(distill_loss),
            "probs_norm": _f(probs_norm),
            "clipped": bool(clipped),
            "train_seconds": _f(train_seconds),
        }
        aux = {k: v for k, v in aux.items() if v is not None}
        self.history.append({"round": r, "n_train": int(self.n_train), **aux})

        conn.send({
            "type": "UPDATE",
            "round": r,
            "client_id": self.client_id,
            "n_train": int(self.n_train),
            "probs": probs,
            "aux": aux,
        })
        log.info("round %d: update sent (local_loss=%s distill_loss=%s probs_norm=%.3f clipped=%s | %.1f s)",
                 r, aux.get("local_loss"), aux.get("distill_loss"),
                 aux.get("probs_norm", 0.0), clipped, train_seconds)

    # ---------------------------------------------------------------- outputs
    def save_final(self, message: dict) -> None:
        """Write this client's copy of the run: final model/consensus + history."""
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            files = []
            if "weights" in message:
                weights = {name: torch.from_numpy(np.asarray(arr, dtype=np.float32))
                           for name, arr in message["weights"].items()}
                self.model.load_state_dict(weights)
                path = self.out_dir / "final_model.pt"
                torch.save(self.model.state_dict(), path)
                files.append(path.name)
            if message.get("targets") is not None:
                path = self.out_dir / "consensus_final.npy"
                np.save(path, np.asarray(message["targets"], dtype=np.float32))
                files.append(path.name)
            if self.history:
                path = self.out_dir / "client_metrics.csv"
                columns = ["round", "n_train", "local_loss", "local_acc", "distill_loss",
                           "delta_norm", "probs_norm", "clipped", "train_seconds"]
                with open(path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                    writer.writeheader()
                    for row in self.history:
                        writer.writerow({k: row.get(k, "") for k in columns})
                files.append(path.name)
            summary = {
                "client_id": self.client_id,
                "architecture": self.arch,
                "shard_file": self.shard_file,
                "n_train": int(self.n_train),
                "rounds_done": len(self.history),
                "final_test_acc": message.get("final_test_acc"),
                "files": files,
            }
            (self.out_dir / "client_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
            log.info("saved client artifacts: %s", self.out_dir)
        except Exception as exc:
            log.warning("could not save client artifacts: %s", exc)

    # ------------------------------------------------------------ connection
    def connected_loop(self) -> None:
        log.info("connecting to %s:%d ...", self.host, self.port)
        conn = protocol.connect(self.host, self.port, self.key, self.connect_timeout_s)
        try:
            conn.send({
                "type": "HELLO",
                "proto": protocol.PROTO_VERSION,
                "client_id": self.client_id,
                "n_train": int(self.n_train),
                "ready": True,
                "arch": self.arch,
            })
            welcome = conn.recv()
            if welcome.get("type") != "WELCOME":
                raise protocol.ProtocolError(f"expected WELCOME, got {welcome.get('type')!r}")
            expected_public = int(welcome.get("public_size", -1))
            if expected_public >= 0 and expected_public != int(self.public_X.shape[0]):
                raise SystemExit(
                    f"public set mismatch: server expects {expected_public} samples, "
                    f"your '{self.public_file}' has {int(self.public_X.shape[0])}; "
                    "re-extract your bundle so all data files come from the same build")
            strategy = str(welcome.get("strategy", "logits"))
            shared = welcome.get("shared_model")
            if strategy == "fedavg" and shared and self.arch != shared:
                raise SystemExit(
                    f"the server runs FedAvg with the shared model '{shared}'; you are '{self.arch}'. "
                    f"Set \"architecture\": \"{shared}\" in client/config.json for FedAvg runs.")
            log.info("connected: strategy=%s | %s rounds planned, public set %d samples, %d classes",
                     strategy, welcome.get("total_rounds"), expected_public,
                     int(welcome.get("num_classes", 10)))
            while True:
                message = conn.recv()
                mtype = message.get("type")
                if mtype == "ROUND":
                    self.handle_round(message, conn)
                elif mtype == "FINISHED":
                    self.save_final(message)
                    log.info("run finished; exiting")
                    return
                else:
                    log.warning("ignoring unexpected message type from server: %r", mtype)
        finally:
            conn.close()

    def run(self) -> None:
        while True:
            try:
                self.connected_loop()
                return  # FINISHED received; clean exit
            except KeyboardInterrupt:
                log.info("interrupted; exiting")
                return
            except SystemExit:
                raise
            except Exception as exc:
                log.warning("connection problem (%s: %s); reconnecting in %.0f s",
                            type(exc).__name__, exc, self.retry_delay_s)
                time.sleep(self.retry_delay_s)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Federated learning client")
    parser.add_argument("--config", default="client/config.json")
    parser.add_argument("--server-host", default=None, dest="server_host")
    parser.add_argument("--server-port", type=int, default=None, dest="server_port")
    parser.add_argument("--client-id", default=None, dest="client_id")
    parser.add_argument("--shard", default=None)
    parser.add_argument("--architecture", default=None)
    args = parser.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.server_host is not None:
        cfg["server_host"] = args.server_host
    if args.server_port is not None:
        cfg["server_port"] = args.server_port
    if args.client_id is not None:
        cfg["client_id"] = args.client_id
    if args.shard is not None:
        cfg["shard_file"] = args.shard
    if args.architecture is not None:
        cfg["architecture"] = args.architecture
    Client(cfg).run()


if __name__ == "__main__":
    main()
