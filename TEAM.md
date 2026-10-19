# TEAM — Federated Learning: IID vs non-IID (Fashion-MNIST)

**What we will work on.** A real federated learning system across our laptops for
the ML exam: one server, three clients. Each client trains the same small CNN on
its own private shard and sends only the trained weights to the server; the
server averages them (FedAvg) and sends the updated model back. Same dataset and
model, two runs (**IID** split and **non-IID** split), a short comparison, and a
live demo that includes a client's network dying mid-round.

**Your part.** One client process each (`START_HERE.txt` says whether you are
`c1` or `c2`). The server and client `c3` run on another team machine; the server
address comes in the group chat.

## What is already set up

- Repo with all the code: shared model, server, a working **reference client**,
  and tools. Plain CPU, no GPU needed.
- **Data**: Fashion-MNIST downloaded, cleaned, normalized, split. Your kit has
  your two shards: `data/processed/iid/<you>.npz` and
  `data/processed/noniid/<you>.npz` (20,000 samples each). The two experiments
  only swap this file; everything else stays identical.
- **The pathway**: the TCP protocol is frozen (`PROTOCOL.md`), and every message
  between machines is encrypted with the shared `secret.key` in your kit. Keep
  that key private: not on GitHub, not in public chats.
- End-to-end checks already pass locally, including a killed client and its
  rejoin.

## What you should do

1. Set up (Windows, from the repo root):
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
2. Check your machine:
   `.venv\Scripts\python tools\bench.py` (your training speed) and
   `.venv\Scripts\python tools\protocol_selftest.py --mode all` (expect
   `RESULT: PASS`).
3. Your client is yours. Use `client/client.py` as is, adapt it, or rewrite it.
   A rewritten client must match `PROTOCOL.md`; prove it with
   `.venv\Scripts\python tools\protocol_selftest.py --mode server`.
4. Join a run. Put the server address into `client/config.json` (`server_host`),
   then:
   ```
   .venv\Scripts\python -m client.client --config client\config.json
   ```
   Leave it running. After a network drop it reconnects and rejoins by itself.
5. Optional: tune your training via `training_override` in your config (batch
   size, local epochs, learning rate). Tell the group what you used; the
   official runs use the server defaults so the IID/non-IID comparison stays
   fair.
6. Before the demo: be ready to explain one round out loud (weights go down,
   delta goes up, weighted average, timeout when a client is missing). Everyone
   presents.

## Fixed for the whole team (change only together)

1. `common/model.py`, the shared CNN: FedAvg averages parameter tensors, so
   different models cannot be mixed.
2. `PROTOCOL.md`, the wire format: one server, one decoder.

Everything else in your client is your call. Anything unclear goes to the group
chat.
