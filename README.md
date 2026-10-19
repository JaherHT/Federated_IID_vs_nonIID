# Federated Learning: IID vs non-IID (Fashion-MNIST)

Machine Learning course project: a federated system running across several
machines on the local network:

```
                       SERVER  (global model, aggregation, evaluation)
                          |
            --------------+--------------
            |             |             |
         client c1     client c2     client c3
         private       private       private
         shard         shard         shard
            |             |             |
            ---- FedAvg aggregation -----
```

- One small CNN (20,490 parameters, ~80 KB per update) shared by all machines.
- **FedAvg**: each round, clients train on their own shard and send weights to
  the server; the server averages them and broadcasts the new global model.
  Raw data never leaves a machine.
- Everything on the wire is **encrypted** (Fernet, one shared key file).

## Layout

```
common/       shared code, identical everywhere (model, data, protocol)
server/       the federated server (round loop, aggregation, evaluation)
client/       the reference client (adapt or rewrite it; see TEAM.md)
data/         prepare.py + raw/ + processed/ (shards, test set, distributions.json)
tools/        gen_key, bench, protocol_selftest, make_peer_bundle, smoke_local
runs/         outputs of every run (logs, csv, summary, final model)
```

## Requirements

Python >= 3.10 (3.12 recommended). Dependencies: `numpy`, `torch`,
`cryptography`. CPU is enough (the model is tiny).

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
# CPU-only torch is smaller and all this project needs:
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On Linux/macOS use `.venv/bin/pip` etc.

## Quickstart

1. `python data/prepare.py` — downloads Fashion-MNIST, builds every shard.
2. `python tools/gen_key.py` — creates `secret.key` (same file on every machine, never committed).
3. `python tools/smoke_local.py` — full server + 2 clients on this machine; expect `RESULT: PASS`.
4. Real run, server machine: `python -m server.server --config server/config.json`
5. Each client machine (edit `server_host` first):
   `python -m client.client --config client/config_c1.json`

Peer bundles: `python tools/make_peer_bundle.py` writes `bundles/peer_cN.zip`
(code + that peer's shards + `secret.key` + pre-filled config +
`START_HERE.txt`). Send privately; the zips contain the key.

## Running the two experiments

Same dataset, same model, same settings; only the partition changes:

- **IID**: clients read `data/processed/iid/<id>.npz` (default in the shipped configs).
- **non-IID**: switch `shard_file` in each client config to `data/processed/noniid/<id>.npz`.
- Both partitions give each client 20,000 samples; IID is a shuffle, non-IID
  sorts by label (~3 classes per client). Evidence:
  `data/processed/distributions.json`.
- Suggested protocol: 15 rounds, 1 local epoch, Adam lr 1e-3 (the defaults).
  Keep every other knob identical between the two runs.

Comparison material comes out of `runs/<timestamp>/`: `metrics.csv`
(global accuracy per round), `client_metrics.csv` (per-client loss/accuracy/
delta norms/participation), `summary.json`, `model_final.pt`.

## What is exchanged between the machines

- Server -> client: the current global weights + the round's training config.
- Client -> server: `delta` = weights after local training minus the weights it
  just received, plus `n_train` and local metrics. (Weights are sent, not
  gradients; deltas keep the server update simple.)
- Aggregation: `delta_avg = sum(n_k / sum(n)) * delta_k`; `global += delta_avg`.
- A reconnected client receives the full current global model at the next
  round; nothing else needs to be reconciled.
- Never exchanged: raw data, labels, another client's update.

Full details in `PROTOCOL.md`.

## Outputs of every run (`runs/<timestamp>/`)

- `server.log` — full log of the run
- `metrics.csv` — round, wall time, participants, global test loss/accuracy
- `client_metrics.csv` — per client per round: status (ok/miss/dead), loss, accuracy, delta norm
- `summary.json` — final numbers + per-client miss counts
- `model_final.pt` — final global model
- `run_config.json` — full configuration of the run

## LAN notes

- Server machine: allow Python through the Windows firewall (private networks)
  on the server port (5000 by default); note its IP (`ipconfig`) and set it in
  every client config. A DHCP address is fine as long as it doesn't change
  mid-run; pin it if you can.
- All machines must be on the same network and able to reach each other. Many
  campus/guest Wi-Fi networks block device-to-device traffic: test early, and
  keep a phone hotspot as fallback.
- The server accepts reconnections at any time; client identity is the
  `client_id`, never the IP, so reconnections survive a changed address.

## Team notes

- Roles, client ownership, and team contracts: `TEAM.md`.
- Custom client implementations are validated against `PROTOCOL.md` with
  `python tools/protocol_selftest.py --mode server`.
