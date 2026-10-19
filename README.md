# Federated Learning: IID vs non-IID (Fashion-MNIST)

Practical exam project for the Machine Learning course. A real federated
system running across several machines on the local network:

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
- **FedAvg**: every round, clients train a little on their own shard and send
  their update to the server; the server averages and broadcasts the new
  global weights. Raw data never leaves a machine.
- Everything on the wire is **encrypted** (Fernet, one shared key file).
- Optional **client-level differential privacy** (clipping + Gaussian noise +
  epsilon accounting), off by default.

## Layout

```
common/       shared code, identical everywhere (model, data, protocol, privacy)
server/       the federated server (round loop, aggregation, DP, evaluation)
client/       the reference client (peers may adapt or rewrite it; see DELEGATION.md)
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

To distribute the work: `python tools/make_peer_bundle.py` produces
`bundles/peer_c1.zip` etc. — each bundle carries the code, that peer's
shards, `secret.key`, a pre-filled client config and a `START_HERE.txt`.
Hand them out privately over chat/USB (they contain the key).

## Running the two experiments

Same dataset, same model, same settings; only the partition changes (the
exam's central control):

- **IID**: clients read `data/processed/iid/<id>.npz` (default in the shipped configs).
- **non-IID**: switch `shard_file` in each client config to `data/processed/noniid/<id>.npz`.
- Both partitions give each client 20,000 training samples; the IID one is a
  shuffle, the non-IID one sorts by label, so each client ends up with ~3
  classes. Evidence is in `data/processed/distributions.json`.
- Suggested protocol: 15 rounds, 1 local epoch, Adam lr 1e-3 (the defaults).
  Keep every other knob identical between the two runs.

Comparison material comes out of `runs/<timestamp>/`: `metrics.csv`
(global accuracy per round), `client_metrics.csv` (per-client loss/accuracy/
delta norms/participation), `summary.json`, `model_final.pt`.

## What is exchanged between the machines

- Server -> client: the current global weights + the round's training config.
- Client -> server: `delta` = weights after local training minus the weights it
  just received, plus `n_train` and local metrics. (Weights are sent, not
  gradients; deltas make both DP clipping and the server update trivial.)
- Aggregation: `delta_avg = sum(n_k / sum(n)) * delta_k`; `global += delta_avg`.
- A reconnected client receives the full current global model at the next
  round; nothing else needs to be reconciled.
- Never exchanged: raw data, labels, another client's update.

Full details in `PROTOCOL.md`.

## The disconnection test (what to show the professor)

Disable the network adapter of one client (`ncpa.cpl`, or `Disable-NetAdapter -Name "Wi-Fi"`).
What happens, and the log line that proves it:

- *Which client stopped answering*: `client c2 marked OFFLINE (...)` or
  `client c2 did not answer within 30 s; skipped for round N`.
- *What happens to the round*: it completes with the clients that answered
  (`round N done ... | 2 update(s)`); the server never waits forever.
- *Does the server stay alive*: yes; rounds keep advancing, other clients keep
  training.
- *When the client returns*: its client process (still running) reconnects by
  itself: `client c2 reconnected (was marked offline)`; the next round includes
  it again with the latest global weights.

Rehearse it locally: `python tools/smoke_local.py --drop c2` (kill + restart)
and `python tools/smoke_local.py --slow c2` (frozen socket, pure timeout
path). Both print a PASS/FAIL checklist.

## Differential privacy (optional, extra)

DP-FedAvg style, client-level `(epsilon, delta)`:

1. clients clip their update delta to L2 norm <= `clip_norm` (C)
2. the server adds Gaussian noise with std `sigma * s` to the averaged delta,
   where `s = C * max(client share)` is the sensitivity of the weighted mean
3. accounting: analytic Gaussian mechanism per round (Balle & Wang 2018),
   composed over rounds (basic composition; conservative but valid)

Enable it: `python -m server.server --config server/config.json --dp --sigma 4 --clip 1.0`
(all clients automatically start clipping; epsilon lands in `server.log` and `summary.json`).

Privacy/utility reference (15 rounds, delta = 1e-5, basic composition):

| noise multiplier sigma | epsilon (total) |
|------------------------|-----------------|
| 1                      | ~74.6           |
| 2                      | ~34.5           |
| 4                      | ~16.2           |
| 8                      | ~7.7            |

Calibrate `clip_norm` from a plain run first: the `delta_norm` column in
`client_metrics.csv` tells you the natural size of the updates; pick C around
their median so clipping barely bites, then pick sigma for your epsilon target.
Expect visible accuracy loss at sigma >= 1 on this small model: that is the
privacy/utility tradeoff itself, and one short DP experiment with two sigma
values makes a good extra slide.

Honest scope, for the defense: this protects the released per-round aggregate
(for example, against guessing whether a specific client participated). It
does **not** hide individual updates from the server (no secure aggregation),
and epsilon accounting here is basic composition, not a tight moments
accountant.

## Outputs of every run (`runs/<timestamp>/`)

- `server.log` — everything the demo needs to point at
- `metrics.csv` — round, wall time, participants, global test loss/accuracy, epsilon spent
- `client_metrics.csv` — per client per round: status (ok/miss/dead), loss, accuracy, delta norm, clipped
- `summary.json` — final numbers + DP parameters + per-client miss counts
- `model_final.pt` — final global model
- `run_config.json` — full configuration of the run

## LAN notes

- Server machine: allow Python through the Windows firewall (private networks)
  on the server port (5000 by default); note its IP (`ipconfig`) and set it in
  every client config. A DHCP address is fine as long as it doesn't change
  mid-demo; pin it if you can.
- All machines must be on the same network and able to reach each other. Many
  campus/guest Wi-Fi networks block device-to-device traffic: test early, and
  keep a phone hotspot as fallback.
- The server accepts reconnections at any time; client identity is the
  `client_id`, never the IP, so a re-enabled adapter with a new address is fine.

## Team notes

- Peers own their client code and their training overrides — see `DELEGATION.md`
  for the exact split between "yours" and "team contract".
- Custom client implementations can be validated against `PROTOCOL.md` with
  `python tools/protocol_selftest.py --mode server`.
