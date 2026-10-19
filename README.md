# Federated Learning: IID vs non-IID (Fashion-MNIST)

Machine Learning course project: a federated system running across several
machines, over the local network or over the internet:

```
                       SERVER  (consensus, server-side model, evaluation)
                          |
            --------------+--------------
            |             |             |
         client c1     client c2     client c3
         own model     own model     own model
         private       private       private
         shard         shard         shard
            |             |             |
            --- probabilities on a shared public set ---
```

- Every client trains **its own architecture** (presets: cnn, mlp, cnn_deep;
  add your own) on its private shard.
- Each round, clients send only **classification probabilities on a shared
  public set** (2000 label-free images). The server averages them into one
  consensus and sends it back; clients distill from it before the next round.
  Weights never leave any machine; raw data never leaves any machine.
- The scheme is the **federated-distillation family** (FedMD / FedDF style):
  FedAvg averages parameter tensors, so it requires identical models on every
  client; exchanging predictions instead lets each client keep its own
  architecture.
- Everything on the wire is **encrypted** (Fernet, one shared key file).

**Layout:**
```
common/       shared code, identical everywhere (model presets, data, protocol, privacy helpers)
server/       the federated server (round loop, averaging, server-side model, evaluation)
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
   expect `RESULT: PASS`.
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
- Both partitions give each client 20,000 samples; IID is a shuffle, non-IID
  sorts by label (~3 classes per client). Evidence:
  `data/processed/distributions.json`.
- Suggested protocol: 15 rounds (the default) with the default training
  config. Keep every other knob identical between the two runs.

Comparison material comes out of `runs/<timestamp>/`: `metrics.csv`
(server-side model accuracy per round), `client_metrics.csv` (per-client
losses, probabilities norm, participation), `summary.json`, `model_final.pt`,
`consensus_final.npy`.

**What is exchanged between the machines:**
- Server -> client: the consensus probabilities (averaged over clients) plus
  the round's training config.
- Client -> server: the client's probabilities on the public set, `n_train`,
  and a few local scalars (losses, norm, timing).
- Aggregation: `consensus = sum(n_k / sum(n)) * probs_k`; the server trains
  its own model on the public set toward that consensus.
- A reconnected client simply continues: the next ROUND carries the current
  consensus and nothing else needs reconciling.
- Never exchanged: raw data, labels, model weights, another client's update.

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
- `client_metrics.csv` — per client per round: status (ok/miss/dead/bad-update), losses, probability norm, clipping
- `summary.json` — final numbers + per-client miss counts
- `model_final.pt` — final server-side model
- `consensus_final.npy` — final consensus probabilities
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
  `python tools/protocol_selftest.py --mode server`.
