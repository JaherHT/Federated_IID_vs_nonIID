# TEAM: Federated Learning, IID vs non-IID (Fashion-MNIST)

**Goal:** the group activity, working end to end on our four machines.
  - a distributed federated system (one server, four clients; c3 runs on the server machine)
  - an IID run and a non-IID run, measured the same way, compared and explained
  - one session where a client loses its network mid-run: the rest keeps going,
    and the client rejoins by itself when it is back

**Two methods, one system:** the main runs are classic FedAvg - every machine
trains the same shared model and sends only weight deltas; the server averages
them. As a secondary task the same server can run the probabilities exchange
(federated distillation): clients send predictions on a shared public set and
may each use a different architecture.

**How it works (main runs):** each round the server sends the current shared
weights; every client trains on its own shard, sends the delta
(trained - received), and the server averages the deltas into the new global
model. Raw data never leaves a machine; only that machine's update does.

**Roles:**
- **Jaher**: server machine + client c3.
- **Rolando**: client c1.
- **Mayte**: client c2.
- **Darío**: client c4.

**What each role does:**
- **Server + c3:** runs the server for every session, posts the server address
  and the round count in the group chat first, watches the server log, and
  runs client c3 on the same machine.
- **Client c1 / c2 / c4:** run one client process during sessions and keep it
  alive; main runs use the shared model (cnn, already set in your config);
  for the secondary task you may switch to your own preset (cnn | mlp |
  cnn_deep | cnn_wide, or add a model in common/model.py); report their
  per-round numbers.

**What you already have:**
- this repo (same code on every machine)
- your two shards, 15,000 samples each: data\processed\iid\ and data\processed\noniid\
- data\processed\public.npz (the shared public set; used by the secondary task)
- secret.key (the protocol key; private, never share it outside the team)
- a pre-filled client\config.json (server address, your shard, shared model)

**Connecting (both ways work; decide per session where everyone is):**
- **Same network (lab, one Wi-Fi):** the server person runs tools\net_info.py and
  posts the printed LAN address (usually 192.168.x.x).
- **Different networks (each of us at home):** install Tailscale on all four
  machines, sign in, join one shared network; the server person then runs
  tools\net_info.py and posts the printed 100.x address. The system itself does
  not change; only the address in server_host does.
- **Everyone else:** put that address into "server_host" in client\config.json.
  Port stays 5000. Campus Wi-Fi is sometimes blocked between devices (test it);
  a phone hotspot is the fallback.

**Client steps (c1, c2, c4):**
1. **Setup (Windows, repo root):**
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
2. **Self-check (expect RESULT: PASS):**
   .venv\Scripts\python tools\bench.py
   .venv\Scripts\python tools\protocol_selftest.py --mode all
3. **Join a session only after the self-check passes:**
   .venv\Scripts\python -m client.client --config client\config.json
   Keep it running; it reconnects by itself after network drops.
4. **Optional:** training_override in your config (batch size, epochs, learning
   rate). Official runs use the server defaults.

**Rules:**
- **The connection carries only this activity:** the server accepts a HELLO
  handshake, then one UPDATE per round (your weight delta in main runs, your
  probabilities in the secondary task). Anything else is refused and the
  connection is closed. No file access and no remote commands, in either
  direction. Full spec: PROTOCOL.md.
- **Main runs need one shared model:** keep "architecture": "cnn" in your
  config. A client with a different model refuses to join a FedAvg session
  (clear error on screen).
- Keep secret.key out of chats, screenshots and git.
- **Fixed team-wide:** PROTOCOL.md, the data files, and the public set.
  Everything else in your client is yours (secondary-task architecture
  included).
- **Everyone must be able to explain:** what is exchanged in each method, how
  the averaging works, what changes between IID and non-IID, and what happens
  when a client drops out.
