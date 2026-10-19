# Federated Learning: IID vs non-IID (Fashion-MNIST)

Machine Learning course project: a federated system running across several
machines, over the local network or over the internet:

```
                       SERVER  (aggregation, server-side model, evaluation)
                          |
            --------------+------------------------
            |             |             |         |
         client c1     client c2     client c3  client c4
         own model     own model     own model  own model
         private       private       private    private
         shard         shard         shard      shard
            |             |             |         |
            --- two exchange modes: probabilities (main) or deltas (FedAvg) ---
```

- Every client trains **its own architecture** in the main mode (presets: cnn,
  mlp, cnn_deep, cnn_wide; add your own) on its private shard.
- **Main scheme (probabilities).** Each round, clients send only
  classification probabilities on a shared public set (2000 label-free
  images). The server averages them into one consensus and sends it back;
  clients distill from it before the next round. Weights never leave any
  machine; raw data never leaves any machine. This is the federated-
  distillation family (FedMD / FedDF style) - FedAvg averages parameter
  tensors, so it requires identical models, and we want different ones.
- **Secondary scheme (FedAvg).** Set `"strategy": "fedavg"` in
  `server/config.json` and every client uses the same model
  (`"architecture": "cnn"`). Clients then send weight deltas, the server
  averages them into the global model, and every client trains from it the
  next round. Same rounds, same evaluation, same reporting - two extra
  comparison rows at the end.
- Everything on the wire is **encrypted** (Fernet, one shared key file).

**Layout:**
```
common/       shared code, identical everywhere (model presets, data, protocol, privacy helpers)
server/       the federated server (round loop, aggregation, server-side model, evaluation)
client/       the reference client (adapt or rewrite it; see TEAM.md)
data/         prepare.py + raw/ + processed/ (shards, public set, eval set, distributions.json)
tools/        gen_key, bench, protocol_selftest, make_peer_bundle, smoke_local, net_info
runs/         outputs of every run (logs, csv, summary, models)
```

**Requirements:** Python >= 3.10 (3.12 recommended). Dependencies: `numpy`,
`torch`, `cryptography`. CPU is enough (the models are small).

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
# CPU-only torch is smaller and all this project needs:
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On Linux/macOS use `.venv/bin/pip` etc.

**Quickstart:**
1. `python data/prepare.py` — downloads Fashion-MNIST, builds every shard,
   the public set and the eval set.
2. `python tools/gen_key.py` — creates `secret.key` (same file on every
   machine, never committed).
3. `python tools/smoke_local.py` — full server + 2 clients on this machine;
   expect `RESULT: PASS`. Add `--strategy fedavg` to smoke-test the FedAvg
   path too.
4. Real run, server machine: `python -m server.server --config server/config.json`
5. Each client machine (edit `server_host` first):
   `python -m client.client --config client/config.json`

Peer bundles: `python tools/make_peer_bundle.py` writes `bundles/peer_cN.zip`
(code + that peer's shards + public set + `secret.key` + pre-filled config +
`START_HERE.txt`). Send privately; the zips contain the key.

**Running the two experiments:** same dataset, same settings, same clients;
only the partition changes.

- **IID**: clients read `data/processed/iid/<id>.npz` (default in the shipped configs).
- **non-IID**: switch `shard_file` in each client config to `data/processed/noniid/<id>.npz`.
- Both partitions give each client 15,000 samples; IID is a shuffle, non-IID
  sorts by label (2-3 classes per client). Evidence:
  `data/processed/distributions.json`.
- Main runs: 15 rounds (the default) with the default training config, strategy
  `logits`. Keep every other knob identical between the two runs.
- Secondary runs: the same pair again with `"strategy": "fedavg"` and
  `"architecture": "cnn"` on every client. Compare methods as well as
  partitions (IID vs non-IID under each strategy).

Comparison material comes out of `runs/<timestamp>/`: `metrics.csv`
(server-model accuracy per round), `client_metrics.csv` (per-client losses,
update norm, participation), `summary.json`, `model_final.pt`, and
`consensus_final.npy` (probabilities mode only).

**What is exchanged between the machines:** two modes, one protocol.

- Probabilities mode: server -> consensus probabilities + round config;
  client -> probabilities on the public set, `n_train`, local scalars.
  Aggregation: `consensus = sum(n_k / sum(n)) * probs_k`; the server trains
  its own model toward that consensus.
- FedAvg mode: server -> the shared weights + round config; client -> weight
  delta (`trained - received`), `n_train`, local scalars. Aggregation:
  `global += sum(n_k / sum(n)) * delta_k`.
- A reconnected client simply continues: the next ROUND carries the current
  state (consensus or weights) and nothing else needs reconciling.
- Never exchanged: raw data, labels, another client's update.

Full details in `PROTOCOL.md`.

**What the connection allows:** the server accepts exactly one conversation:
a HELLO handshake, then one UPDATE per round. Everything else — another
client id, a not-ready client, unknown messages, malformed fields — is
refused and the connection is closed (each refusal is logged with its
reason). Payloads are plain data (JSON + float32 arrays): nothing received
over the network is ever executed, imported or written to disk.
`python tools/protocol_selftest.py --mode probe` proves these refusals
against a running server.

**Outputs of every run (`runs/<timestamp>/`):**
- `server.log` — full log of the run (including every disconnect/rejoin)
- `metrics.csv` — round, wall time, participants, server-model test loss/accuracy
- `client_metrics.csv` — per client per round: status (ok/miss/dead/bad-update), losses, update norm, clipping
- `summary.json` — final numbers, strategy, per-client miss counts
- `model_final.pt` — final server-side model (the global model in FedAvg mode)
- `consensus_final.npy` — final consensus probabilities (probabilities mode only)
- `run_config.json` — full configuration of the run

**Networks:**
- Server machine: allow Python through the Windows firewall on the server port
  (5000 by default). `python tools/net_info.py` prints the address to share.
- Same network (lab, one Wi-Fi): share the printed LAN address (192.168.x.x or
  similar). A DHCP address is fine as long as it doesn't change mid-run.
- Different networks (everyone at home): install Tailscale (or a similar mesh
  VPN) on all four machines, join one shared network, and share the printed
  100.x address instead. Nothing else changes; the address only goes into
  `server_host`.
- Campus/guest Wi-Fi sometimes blocks device-to-device traffic: test early,
  keep a phone hotspot as fallback.
- The server accepts reconnections at any time; client identity is the
  `client_id`, never the IP, so reconnections survive a changed address.

**Team notes:**
- Roles, duties, session steps: `TEAM.md`.
- Custom client implementations are validated against `PROTOCOL.md` with
  `python tools/protocol_selftest.py --mode server` (it exercises both the
  probabilities round and the FedAvg delta round).
