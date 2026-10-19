# TEAM: Federated Learning, IID vs non-IID (Fashion-MNIST)

One server, three clients. Each client trains the shared CNN on its own shard
and sends only weights; the server averages them (FedAvg). Two runs to compare
(IID and non-IID), plus a live demo where a client loses its network mid-run.

Roles:
- [name]: server machine + client c3.
- [name]: client c1.
- [name]: client c2.

What each role does:
- Server + c3: runs the federation from the server machine and posts the server
  address to the group before a session. Client c3 runs on the same machine.
- Client c1 / c2: runs one client process during sessions and keeps it alive;
  owns its client code (the reference client as is, or a rewrite that matches
  PROTOCOL.md and passes tools\protocol_selftest.py --mode server); reports its
  per-round numbers.

What you already have: this repo; your two shards (data\processed\iid\... and
data\processed\noniid\..., 20,000 samples each); secret.key (private; required
to connect); a pre-filled client\config.json.

Client steps (c1, c2):
1. Setup (Windows, repo root):
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
2. Self-check (expect RESULT: PASS):
   .venv\Scripts\python tools\bench.py
   .venv\Scripts\python tools\protocol_selftest.py --mode all
3. Run: set "server_host" in client\config.json (address from the group chat), then:
   .venv\Scripts\python -m client.client --config client\config.json
   Keep it running; it reconnects by itself after network drops.
4. Optional: training_override in your config (batch size, local epochs,
   learning rate). Official runs use the server defaults.

Fixed team-wide: common/model.py (FedAvg needs identical models) and PROTOCOL.md
(one server, one decoder). Everything else in your client is yours.
