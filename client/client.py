"""Reference federated learning client.

This file is a *reference*: you can copy it, adapt it, or rewrite it. The
contract is PROTOCOL.md, not this code. Two things must match the rest of the
team: the model topology (common/model.py) and the message format.

What one round looks like, from the client side:
  1. receive ROUND (current global weights + training config)
  2. train locally for the given number of epochs on your own shard
  3. delta = weights after training - weights before training
  4. if the round config carries a clip_norm (DP on), clip the delta to that
     L2 norm before sending
  5. send UPDATE with the delta + your local metrics

On any network error: log, close, wait retry_delay_s, reconnect and continue.
The server accepts reconnections at any time; you always receive the latest
global weights at the start of the next round, so nothing has to be reconciled.

Run from the repo root:

    python -m client.client --config client/config_c1.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):  # allow `python client/client.py` as well
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data as data_mod
from common import privacy
from common import protocol
from common.model import make_model

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
log = logging.getLogger("client")


def snapshot(model) -> dict:
    return {k: v.detach().cpu().numpy().astype("float32") for k, v in model.state_dict().items()}


def train_local(model, shard_path: str, cfg: dict) -> float:
    """Local training step: a few epochs of SGD/Adam on the local shard.

    Returns the mean training loss of the last epoch.
    """
    batch_size = int(cfg.get("batch_size", 64))
    epochs = int(cfg.get("local_epochs", 1))
    lr = float(cfg.get("lr", 0.001))
    optimizer_name = str(cfg.get("optimizer", "adam")).lower()

    loader = data_mod.make_loader(shard_path, batch_size=batch_size, shuffle=True)
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        last_loss = epoch_loss / max(batches, 1)
    return last_loss


class Client:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client_id = str(cfg["client_id"])
        self.host = str(cfg["server_host"])
        self.port = int(cfg["server_port"])
        self.shard = data_mod.Shard(cfg["shard_file"])
        self.model = make_model()
        self.retry_delay = float(cfg.get("retry_delay_s", 3))
        self.connect_timeout = float(cfg.get("connect_timeout_s", 10))
        self.local_metrics = bool(cfg.get("local_metrics", True))
        self.override = cfg.get("training_override") or {}

        enc = cfg.get("encryption", {})
        self.key = protocol.load_key(enc.get("key_file", "secret.key")) if enc.get("enabled", True) else None

    def setup_logging(self) -> None:
        fmt = logging.Formatter(LOG_FORMAT)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        handlers = [stream]
        if self.cfg.get("log_file"):
            file_handler = logging.FileHandler(self.cfg["log_file"], encoding="utf-8")
            file_handler.setFormatter(fmt)
            handlers.append(file_handler)
        root.handlers[:] = handlers

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        self.setup_logging()
        counts = self.shard.class_counts()
        nonzero = " ".join(f"{c}:{n}" for c, n in counts.items() if n)
        log.info("client %s ready | shard=%s | %d samples", self.client_id, self.shard.path, len(self.shard))
        log.info("class counts: %s", nonzero)
        while True:
            try:
                self.connected_loop()
                return  # clean shutdown (FINISHED from the server)
            except KeyboardInterrupt:
                log.info("interrupted by user; exiting")
                return
            except Exception as exc:
                log.warning("connection problem (%s: %s); retrying in %.0f s ...",
                            type(exc).__name__, exc, self.retry_delay)
                time.sleep(self.retry_delay)

    def connected_loop(self) -> None:
        conn = protocol.connect(self.host, self.port, self.key, self.connect_timeout)
        log.info("connected to %s:%d", self.host, self.port)
        conn.send({"type": "HELLO", "proto": protocol.PROTO_VERSION,
                   "client_id": self.client_id, "n_train": len(self.shard)})
        welcome = conn.recv()
        log.info("handshake ok (server announced %s total rounds)", welcome.get("total_rounds"))
        while True:
            message = conn.recv()
            mtype = message.get("type")
            if mtype == "FINISHED":
                log.info("server finished the run; exiting")
                conn.close()
                return
            if mtype != "ROUND":
                log.warning("ignoring unknown message type %r", mtype)
                continue
            self.handle_round(conn, message)

    def handle_round(self, conn, message: dict) -> None:
        r = int(message["round"])
        cfg = dict(message.get("config", {}))
        cfg.update(self.override)  # a peer's own tuning wins over server defaults

        self.model.load_state_dict({k: torch.from_numpy(v) for k, v in message["weights"].items()})
        before = snapshot(self.model)

        t0 = time.time()
        train_loss = train_local(self.model, self.shard.path, cfg)
        after = snapshot(self.model)
        delta = {k: (after[k] - before[k]).astype("float32") for k in before}

        delta_norm = privacy.l2_norm(delta)
        clipped = False
        clip_norm = cfg.get("clip_norm")
        if clip_norm is not None:
            delta, clipped, delta_norm = privacy.clip_delta(delta, float(clip_norm))

        local_acc = 0.0
        if self.local_metrics:
            _, local_acc = data_mod.evaluate(self.model, self.shard.path)

        # debug hook used by tools/smoke_local.py to simulate a frozen network:
        # hold this round's update for sleep_s seconds before sending it.
        delay = self.cfg.get("debug_delay")
        if isinstance(delay, dict) and int(delay.get("round", -1)) == r:
            seconds = float(delay.get("sleep_s", 30.0))
            log.info("debug hook: holding the round-%d update for %.0f s (simulated network freeze)", r, seconds)
            time.sleep(seconds)

        conn.send({
            "type": "UPDATE",
            "round": r,
            "client_id": self.client_id,
            "n_train": len(self.shard),
            "delta": delta,
            "aux": {
                "train_loss": float(train_loss),
                "local_acc": float(local_acc),
                "delta_norm": float(delta_norm),
                "clipped": bool(clipped),
                "train_seconds": float(time.time() - t0),
            },
        })
        log.info("round %d: train_loss=%.4f | local_acc=%.4f | |delta|=%.3f%s | %.1f s%s",
                 r, train_loss, local_acc, delta_norm,
                 " [clipped]" if clipped else "", time.time() - t0,
                 " [held by debug hook]" if isinstance(delay, dict) and int(delay.get("round", -1)) == r else "")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Federated learning client (reference implementation)")
    parser.add_argument("--config", required=True, help="path to a client config json")
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    Client(cfg).run()


if __name__ == "__main__":
    main()
