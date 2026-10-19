# Delegation: roles, ownership, and the rules that keep the federation working

Team of 3, four roles from the brief (server + 3 clients):

| Role        | Who        | Runs on their laptop          |
|-------------|------------|-------------------------------|
| Server      | [you]      | server process + client `c3`  |
| Client `c1` | [Peer 1]   | one client process            |
| Client `c2` | [Peer 2]   | one client process            |

Edit the names above to the real ones before sharing this file.

## What each peer receives (zip bundle from `tools/make_peer_bundle.py`)

- full repo copy (code + docs; no raw data, no other peers' shards, no .git)
- `secret.key` — the same key file for everyone; keep it out of git and out of public channels
- their shards for BOTH scenarios: `data/processed/iid/<id>.npz` and `data/processed/noniid/<id>.npz`
  (download, cleaning, splitting and normalization constants are already done;
  treat these files as read-only)
- `client/config.json` pre-filled with their id and shard paths — only `server_host` needs editing
- `START_HERE.txt` — exact setup + run commands

## Yours to decide (no coordination needed)

- **Your client code.** Adapt `client/client.py` or rewrite it completely. The
  contract is `PROTOCOL.md`, not the reference file. Prove conformance with
  `tools/protocol_selftest.py` before joining a group run.
- **Your local training settings.** The server sends training defaults in every
  ROUND (`local_epochs`, `batch_size`, `lr`, `optimizer`). You may override any
  of them in your client config under `"training_override"`, e.g.
  `{"local_epochs": 2}`. Use it deliberately and note it in the logs; the
  official IID/non-IID comparison runs use the server defaults.
- **Your extra evidence.** Attach anything you like to `UPDATE.aux`; the server
  records it in `client_metrics.csv`.

## Team contracts (do NOT change on one machine alone)

1. **Model topology** — `common/model.py`. FedAvg averages parameter tensors
   elementwise, so client and server must hold the exact same architecture with
   the exact same keys. A model change is a team-wide edit: one person changes
   the file, everyone pulls, everyone re-runs.
2. **Wire format** — `PROTOCOL.md` (framing, encryption, message fields). One
   server, one decoder: your sending side must produce exactly these frames.
3. **Shard files** — read-only. The entire IID vs non-IID comparison rests on
   everyone training on the split they were given.
4. **Required UPDATE fields** (`round`, `client_id`, `n_train`, `delta`) and
   **clipping behavior** when `clip_norm` is present in the round config. The
   aggregate's privacy guarantee depends on every client clipping.

## Checklist per peer

1. Install Python 3.12 + dependencies (see `START_HERE.txt`).
2. `python tools/bench.py` — see how fast your laptop trains (client = the
   only compute-heavy role; if your numbers are slow, tell the group).
3. `python tools/protocol_selftest.py --mode all` — must print PASS.
4. Only if you rewrote the client: `python tools/protocol_selftest.py --mode server`
   and connect your client to `127.0.0.1:5001` — must print PASS.
5. Join the group run (server address comes from the group chat).
6. Bring evidence to the next meeting: your client logs (per-round train loss,
   local accuracy), anything odd you saw, any overrides you used.

## FAQ

- **Can I use a different model?** Only as a team-wide change (reason above).
  You can experiment locally, but anything that joins the federation must match
  `common/model.py`.
- **Can I skip the clipping step?** When the server has DP on, the ROUND config
  carries `clip_norm` and clients must clip before sending. Keep that part of
  the reference implementation (or re-implement it exactly).
- **My laptop is slow.** `tools/bench.py` gives your per-epoch time. Adjust
  `batch_size` in `training_override`, and say so — official runs use server
  defaults.
- **I got disconnected during a demo?** Nothing to do: keep the process
  running. The client retries the connection automatically every
  `retry_delay_s` and rejoins at the next round with the latest global model.
