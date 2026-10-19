# TEAM: Federated Learning, IID vs non-IID (Fashion-MNIST)

We are building the federated learning system for the ML exam as a real
federation across our machines: one server, three clients. Every client trains
the same small CNN on its own private shard and sends only the trained weights
to the server; the server averages them (FedAvg) and sends the updated model
back. Same dataset and model, two runs (IID split and non-IID split), a short
comparison, and a live demo that includes a client's network dying mid-round.
You each take one client process (`START_HERE.txt` says whether you are `c1` or
`c2`); the server and client `c3` run on another team machine, and the server
address comes in the group chat.

Everything needed to start is already in place, so nothing has to be rebuilt.
The repo carries the whole system: the shared model, the server, a working
reference client, and the tooling, all CPU-only. The data is done as well:
Fashion-MNIST cleaned, normalized, and split, with your two shards in your kit
(`data/processed/iid/...` for the IID run and `data/processed/noniid/...` for
the non-IID run, 20,000 samples each; the two runs only swap this file). The
communication pathway is frozen too: the TCP protocol, framing, and encryption
are specified in `PROTOCOL.md`, and the shared `secret.key` in your kit makes
the link work. Keep that key private: not on GitHub, not in public chats. The
full loop is already verified end to end, including a killed client and its
rejoin.

Your side of the work, in a nutshell:

1. Set up (Windows, from the repo root):
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
2. Check your machine: `.venv\Scripts\python tools\bench.py` (your training
   speed) and `.venv\Scripts\python tools\protocol_selftest.py --mode all`
   (expect `RESULT: PASS`).
3. Your client is yours: use `client/client.py` as is, adapt it, or rewrite it.
   A rewritten client must match `PROTOCOL.md`; prove it with
   `.venv\Scripts\python tools\protocol_selftest.py --mode server`.
4. Join a run: put the server address into `client/config.json` (`server_host`),
   then launch with `.venv\Scripts\python -m client.client --config
   client\config.json` and leave it running. After a network drop it reconnects
   and rejoins by itself.
5. Optional: tune your training via `training_override` in your config (batch
   size, local epochs, learning rate). Tell the group what you used; the
   official runs use the server defaults so the IID/non-IID comparison stays
   fair.

Two things stay identical across the whole team, because FedAvg averages
parameter tensors elementwise and one server decodes one format: the model in
`common/model.py` and the wire format in `PROTOCOL.md`. Everything else in your
client is your call. Be ready to explain one round out loud before the demo
(weights go down, delta goes up, weighted average, timeout when someone is
missing), and bring anything unclear to the group chat.
