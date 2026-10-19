# Wire protocol v1

Transport: plain TCP, one connection per client, initiated by the client.
The reference implementation lives in `common/protocol.py`; this document is
the contract. A client may be re-implemented, but it must match this spec
byte for byte.

**Framing:** every message is a single frame.

```
4 bytes   payload length N   (unsigned, big-endian)
N bytes   payload
```

Payload = zlib-compressed UTF-8 JSON, then wrapped in a Fernet token.
Maximum frame: 16 MB. Maximum size after decompression: 64 MB.

**Serialization (safe by construction):** the payload is plain data. There is
no pickle, no eval, no imports, no file paths and no commands anywhere in this
protocol; the receiving side rejects everything that is not in this list:

- JSON values: objects, arrays, strings, integers, finite floats, booleans, null
- float32 numpy arrays, carried as
  `{"__ndarray__": {"dtype": "float32", "shape": [...], "b64": "..."}}`
  where `b64` is the raw little-endian bytes, base64-encoded

A float32 array is the only binary type on the wire. A typical UPDATE is
2000 x 10 probabilities (~80 KB raw, less after zlib). Non-finite numbers
(inf/nan) are rejected on both the sending and the receiving side.

**Encryption:**
- Fernet (`cryptography` package): AES-128-CBC + HMAC-SHA256, symmetric.
- One shared key file (`secret.key`, produced by `tools/gen_key.py`) identical
  on every machine. Never commit it to git.
- Both ends must use the same key; a wrong key shows up as `InvalidToken` on
  the first message (and the server drops the connection).

**Messages:** all messages are dicts with a `"type"` field. Every field listed
below is validated by the server; a message that fails validation closes the
connection (see "What the server accepts").

**HELLO (client -> server; first message on every connection):**

| field     | type | notes                                                       |
|-----------|------|-------------------------------------------------------------|
| type      | str  | "HELLO"                                                     |
| proto     | int  | protocol version, currently 1                               |
| client_id | str  | stable identity, 1..64 chars, e.g. "c1"; reconnects reuse it|
| n_train   | int  | local training samples, 1..10,000,000 (aggregation weight)  |
| ready     | bool | must be true: model and data loaded before connecting       |
| arch      | str  | short architecture name, 1..64 chars, e.g. "cnn"            |

**WELCOME (server -> client):**

| field       | type | notes                       |
|-------------|------|-----------------------------|
| type        | str  | "WELCOME"                   |
| proto       | int  | protocol version            |
| total_rounds| int  | rounds in the run           |
| public_size | int  | samples in the public set   |
| num_classes | int  | 10                          |

The client checks `public_size` against its own public file and refuses to
join on a mismatch (all data files must come from the same build).

**ROUND (server -> client; start of every round):**

| field    | type            | notes                                               |
|----------|-----------------|-----------------------------------------------------|
| type     | str             | "ROUND"                                             |
| round    | int             | 1-based round number                                |
| targets  | ndarray or null | consensus probabilities (public_size, 10) float32; null in round 1 |
| config   | dict            | local_epochs, kd_epochs, temperature, batch_size, lr, optimizer; `clip_norm` when clipped updates are requested |

**UPDATE (client -> server; answer to a ROUND):**

| field     | type    | notes                                                        |
|-----------|---------|--------------------------------------------------------------|
| type      | str     | "UPDATE"                                                     |
| round     | int     | must equal the round being answered                          |
| client_id | str     | same id as in HELLO                                          |
| n_train   | int     | local sample count                                           |
| probs     | ndarray | float32 (public_size, 10), values in [0, 1], rows sum ~= 1   |
| aux       | dict    | scalars only; reference fields: local_loss, local_acc, distill_loss, probs_norm, clipped, train_seconds |

**FINISHED (server -> client):** sent once after the last round. The client
should exit cleanly.

**Round semantics:**
1. The server broadcasts ROUND `r` to every connected client. The clients to
   which the send succeeded form the "expected" set for the round.
2. Each expected client, in order:
   - trains on its local shard (cross-entropy, `local_epochs` passes),
   - if `targets` is present, distills from the consensus on the public set
     (kl-divergence at the given `temperature`),
   - computes softmax probabilities over the public set at that temperature,
   - sends exactly one UPDATE `r`.
3. The server waits until every expected update arrived or `round_timeout_s`
   passes. A client that did not answer is logged and skipped for that round;
   the round still completes. A late update for a closed round is logged and
   ignored. The server never blocks on one client.
4. Aggregation: sample-weighted mean of the received probability matrices
   (`sum over k of (n_k / sum(n)) * probs_k`). The server then trains its
   server-side model on the public set toward that consensus (distillation)
   and evaluates it on the held-out test set.
5. The consensus is broadcast in the next round's `targets`.

**Clipping (when `clip_norm` is configured):** when `config.clip_norm` is
present in ROUND, the client must scale the deviation of its probabilities
from the uniform distribution so its total L2 norm is at most `clip_norm`,
then add the uniform layer back (this keeps every row a valid probability
row). Report the pre-clip norm and whether clipping happened in
`aux.probs_norm` / `aux.clipped`.

**What the server accepts:** exactly this conversation, and nothing else.

- First message on a connection: HELLO with valid fields (known client_id,
  `ready` true, sane n_train). Anything else closes the connection.
- After that: one UPDATE per open round, with valid fields and probabilities
  of the exact expected shape and range. Anything else closes the connection.
- Duplicate and late updates are ignored (not a violation).
- A rejected client may reconnect immediately; the server logs every refusal
  and reason.

There is no message that reads, writes or lists anything on the server
machine, and none that executes anything. `tools/protocol_selftest.py
--mode probe` verifies these refusals against a running server.

**Disconnection and rejoin:**
- A client that misses `round_timeout_s` is skipped for that round but stays
  connected; a client whose connection dies is marked OFFLINE on the next
  failed read/write.
- The server keeps running with the remaining clients and keeps accepting new
  connections at any time, including between and during rounds.
- On any network error the client closes the socket, waits `retry_delay_s`,
  reconnects and sends HELLO again. The server replies WELCOME; the client
  waits for the next ROUND. Nothing needs reconciling: a rejoining client
  simply continues training; the consensus keeps applying to it like to
  everyone else.
- Identity is `client_id`, never the IP address: a disabled/re-enabled network
  adapter may come back with a different IP.

**Versioning:** `proto: 1`. Increment it on any breaking change; the server
logs mismatches and refuses the connection.

**Conformance testing:** `tools/protocol_selftest.py` checks any client
implementation against a two-round test server (HELLO fields, one UPDATE per
round, probability shape/dtype/range, distillation round included):

```
python tools/protocol_selftest.py --mode all      # reference client self-check
python tools/protocol_selftest.py --mode server   # then point YOUR client at 127.0.0.1:5001
python tools/protocol_selftest.py --mode probe    # against a RUNNING server: junk is refused
```
