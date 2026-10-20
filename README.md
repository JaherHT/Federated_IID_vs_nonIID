# Federated Learning: IID vs non-IID (Fashion-MNIST)

Machine Learning course project: a federated system running across several
machines, over the local network or over the internet:

```
                       SERVER  (global model, aggregation, evaluation)
                          |
            --------------+------------------------
            |             |             |         |
         client c1     client c2     client c3  client c4
         shared        shared        shared     shared
         model         model         model      model
         private       private       private    private
         shard         shard         shard      shard
            |             |             |         |
            --- two exchange modes: weight deltas (main) or probabilities ---
```

- **Main scheme (FedAvg).** Each round the server sends the current global
  weights; every client trains on its private shard and sends the delta
  (`trained - received`); the server averages the deltas into the new global
  model and evaluates it. All clients run the same model - the small CNN
  (20,490 parameters, ~80 KB per update). Raw data never leaves a machine.
- **Secondary scheme (probabilities exchange / federated distillation).** Set
  `"strategy": "logits"` in `server/config.json`. Clients then send predictions
  on a shared public set (2000 label-free images); the server averages them
  into one consensus and sends it back, and clients distill from it. Weights
  never move, so each client may run a different architecture (presets: cnn,
  mlp, cnn_deep, cnn_wide; add your own). This is the FedMD / FedDF family -
  useful as a second comparison: predictions vs weight deltas, and
  heterogeneous vs shared models.
- Everything on the wire is **encrypted** (Fernet, one shared key file).

**Layout:**
```
common/       shared code, identical everywhere (model presets, data, protocol helpers)
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
3. `python tools/smoke_local.py` — full server + 2 clients on this machine
   (FedAvg); expect `RESULT: PASS`. Add `--strategy logits` to smoke-test the
   probabilities path too.
4. Real run, server machine: `python -m server.server --config server/config.json`
5. Each client machine (edit `server_host` first):
   `python -m client.client --config client/config.json`

**Peer bundles:** `python tools/make_peer_bundle.py` writes `bundles/peer_cN.zip`
(code + that peer's shards + public set + `secret.key` + pre-filled config +
`START_HERE.txt`). Send privately; the zips contain the key.

**Tools:** every script in `tools/` takes `--help`.
- **gen_key.py:** writes `secret.key`. Run once by the server owner; every
  machine uses that same file.
- **bench.py:** times one round of client work on this machine (local epoch,
  distillation, probabilities, evaluation); `--model all` covers every preset.
  Run it before exam day and share the number with the group.
- **protocol_selftest.py:** conformance check against `PROTOCOL.md`.
  `--mode all` runs a reference client against a local test server (expect
  `RESULT: PASS`); `--mode server` lets a custom client connect to
  `127.0.0.1:5001`; `--mode probe` runs against a live server and proves that
  junk, unknown ids and not-ready clients are refused.
- **make_peer_bundle.py:** builds one zip per peer. Server owner only.
- **smoke_local.py:** full local dry run (one server + two clients).
  `--strategy logits`, `--drop c2`, `--slow c2`, `--rounds N`.
- **net_info.py:** prints the addresses to share (LAN `192.168.x.x` and
  Tailscale `100.x`).

**Running the two experiments:** same dataset, same settings, same clients;
only the partition changes.

- **IID:** clients read `data/processed/iid/<id>.npz` (default in the shipped configs).
- **non-IID:** switch `shard_file` in each client config to `data/processed/noniid/<id>.npz`.
- **Partition sizes:** each client gets 15,000 samples in both partitions; IID
  is a shuffle, non-IID sorts by label (2-3 classes per client). Evidence:
  `data/processed/distributions.json`.
- **Main runs (FedAvg, default):** 15 rounds with the default training config.
  Keep every other knob identical between the two runs.
- **Secondary runs (optional):** the same pair again with `"strategy": "logits"`;
  clients may then use different architectures. Compare methods as well as
  partitions (IID vs non-IID under each strategy).

Comparison material comes out of `runs/<timestamp>/`: `metrics.csv`
(global-model accuracy per round), `client_metrics.csv` (per-client losses,
update norm, participation), `summary.json`, `model_final.pt`, and
`consensus_final.npy` (probabilities mode only).

**What is exchanged between the machines:** two modes, one protocol.

- **FedAvg mode (main):** server -> the shared weights + round config; client ->
  weight delta, `n_train`, local scalars. Aggregation:
  `global += sum(n_k / sum(n)) * delta_k`; the server evaluates the updated
  model on the held-out test set.
- **Probabilities mode:** server -> consensus probabilities + round config;
  client -> probabilities on the public set, `n_train`, local scalars.
  Aggregation: `consensus = sum(n_k / sum(n)) * probs_k`; the server trains
  its own model toward that consensus.
- **Reconnections:** a reconnected client simply continues; the next ROUND
  carries the current state (weights or consensus) and nothing else needs
  reconciling.
- **Never exchanged:** raw data, labels, another client's update.

Full details in `PROTOCOL.md`.

**Timers and limits:**
- **Start window (`start_timeout_s`):** the server waits this long for the full expected set (`expected_client_ids`). On expiry it starts with the clients that are connected, provided there are at least `min_clients` (1); with nobody connected it re-checks every 30 s. A progress line appears every 15 s with the seconds left.
- **Round window (`round_timeout_s`):** each round stays open until every expected client answers or the window closes. Missing or late updates are logged and skipped; the round completes with the rest. A progress line appears every 5 s.
- **Client connect (`connect_timeout_s`):** per-attempt timeout; on failure the client waits `retry_delay_s` and retries forever.
- **Nobody connected:** between rounds the server idles (`no connected clients; waiting for someone to come back`), keeps the port open and accepts reconnections at any time.
- **Mid-round join:** a client that connects during a round is welcomed immediately and joins at the next round; the current round is neither extended nor restarted.
- **End of run:** `FINISHED` carries the final weights (FedAvg) or consensus (probabilities) plus `final_test_acc`; each client writes `client_out/` and exits.

**Configuration reference** (`server/config.json`):
| Key | Shipped value | Meaning |
|---|---|---|
| `network.host` / `network.port` | `0.0.0.0` / `5000` | listen address and port |
| `strategy` | `fedavg` | `fedavg` (main) or `logits` (secondary) |
| `rounds` | `15` | federated rounds |
| `expected_clients` / `expected_client_ids` | `4` / `c1..c4` | who the server waits for; other ids are refused |
| `min_clients` | `1` | below this the run does not start |
| `start_timeout_s` | `30` | start window (code default 300) |
| `round_timeout_s` | `90` | round window |
| `training.*` | `local_epochs 1`, `batch 64`, `lr 1e-3`, `adam` | sent to clients in every ROUND |
| `model.architecture` | `cnn` | shared model for FedAvg; also the server's own model |
| `paths.public_file` / `paths.test_file` / `paths.out_root` | `data/processed/...`, `runs` | inputs and where run folders go |
| `encryption.enabled` / `encryption.key_file` | `true` / `secret.key` | Fernet on every frame |

**Configuration reference** (`client/config.json`):
| Key | Shipped value | Meaning |
|---|---|---|
| `server_host` / `server_port` | `100.70.85.123` / `5000` | pre-filled in your bundle |
| `client_id` | `c1..c4` | must match one of the expected ids |
| `architecture` | `cnn` | FedAvg needs the shared model; probabilities mode allows any preset |
| `shard_file` | `data/processed/iid/<id>.npz` | switch to `noniid/` for the second experiment |
| `public_file` | `data/processed/public.npz` | used by probabilities mode |
| `connect_timeout_s` / `retry_delay_s` | `10` / `3` | per-attempt timeout, pause between retries |
| `training_override` | `null` | optional per-client overrides; official runs use the defaults |
| `log_file` | `null` | optional client log file |
| `out_dir` | `client_out/<id>_<timestamp>` | where the final model and round history are written |

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
- `metrics.csv` — round, wall time, participants, global-model test loss/accuracy
- `client_metrics.csv` — per client per round: status (ok/miss/dead/bad-update), losses, update norm, clipping
- `summary.json` — final numbers, strategy, per-client miss counts
- `model_final.pt` — final global model
- `consensus_final.npy` — final consensus probabilities (probabilities mode only)
- `run_config.json` — full configuration of the run

**Each client's copy:** on `FINISHED` a client writes
`client_out/<client_id>_<timestamp>/` with `final_model.pt` (the final global
weights it received; `consensus_final.npy` in probabilities mode),
`client_metrics.csv` (its round-by-round local loss/accuracy/norm) and
`client_summary.json`. The server's `runs/` stays the authoritative record;
`client_out/` is that machine's own copy of the result.

**Networks:**
- **Server machine:** allow Python through the Windows firewall on the server port
  (5000 by default). `python tools/net_info.py` prints the address to share.
- **Same network (lab, one Wi-Fi):** share the printed LAN address (192.168.x.x or
  similar). A DHCP address is fine as long as it doesn't change mid-run.
- **Different networks (everyone at home):** install Tailscale (or a similar mesh
  VPN) on all four machines, join one shared network, and share the printed
  100.x address instead. Nothing else changes; the address only goes into
  `server_host`.
- **Campus/guest Wi-Fi:** sometimes blocks device-to-device traffic, and some campuses forbid VPNs entirely — in that case use the same-network mode (a phone hotspot or the lab Wi-Fi) and leave Tailscale for off-campus. Test early; keep a hotspot as fallback.
- **Reconnections:** the server accepts them at any time; client identity is the
  `client_id`, never the IP, so reconnections survive a changed address.

**Troubleshooting:**
- **Port test False while the server prints `listening`:** wrong address, missing ACL grant, or the firewall rule is off. Run `tailscale ping` first, then `Test-NetConnection`; a working test shows `InterfaceAlias: Tailscale` and `SourceAddress: 100.x`.
- **`bad handshake (connection closed by peer)` right after a port test:** normal — the test opens TCP and closes without a HELLO.
- **`refused connection ... expected HELLO first`:** normal for probes and port tests.
- **`unknown client_id`:** your `client_id` is not in `expected_client_ids` (`c1..c4`).
- **`client not ready` or a public-size mismatch:** stale bundle or edited data; unzip the current bundle fresh.
- **Architecture mismatch in FedAvg:** the client refuses with a clear message; main runs use `cnn` everywhere.
- **`no connected clients; waiting for someone to come back`:** normal after a full drop; the server keeps listening.
- **Slow rounds:** `tools/bench.py` gives your per-round time; typical full-size rounds take 15-30 s.

**Team notes:**
- Roles, duties, session steps: `TEAM.md`.
- Custom client implementations are validated against `PROTOCOL.md` with
  `python tools/protocol_selftest.py --mode server` (it exercises both the
  FedAvg delta round and the probabilities rounds).

**Sharing this repository:** the GitHub repository carries code and docs only —
no shards, no `secret.key`, no run outputs, no private notes. Your shard and
the key arrive in your private bundle; never commit `secret.key`. If a bundle
leaks, say so in the group and the key gets rotated for everyone.
