# Wire protocol v1

Transport: plain TCP, one connection per client, initiated by the client.
The reference implementation lives in `common/protocol.py`; this document is
the contract. A client may re-implement the sending side, but it must match
this spec byte for byte.

## Framing

Every message is a single frame:

```
4 bytes   payload length N   (unsigned, big-endian)
N bytes   payload
```

Payload = zlib-compressed pickle (protocol 2+) of a Python dict, then wrapped
in a Fernet token. Maximum frame: 200 MB.

Terminology: "weights" and "delta" are dicts `{parameter_name: float32 numpy array}`
using PyTorch `state_dict` keys (e.g. `0.weight`, `0.bias`, ...).

## Encryption

- Fernet (`cryptography` package): AES-128-CBC + HMAC-SHA256, symmetric.
- One shared key file (`secret.key`, produced by `tools/gen_key.py`) identical on
  every machine. Never commit it to git.
- Both ends must use the same key; a mismatch shows up as `InvalidToken` on the
  first message.
- For debugging only, encryption can be disabled with `"encryption": {"enabled": false}`
  on both sides (framing stays the same; payload is the raw compressed pickle).

## Messages

All messages are dicts with a `"type"` field. Unknown extra fields are ignored
(forward compatibility); senders may attach extras freely.

### HELLO  (client -> server; first message on every connection)

| field       | type | notes                                                |
|-------------|------|------------------------------------------------------|
| type        | str  | "HELLO"                                              |
| proto       | int  | protocol version, currently 1                        |
| client_id   | str  | stable identity, e.g. "c1"; reconnects reuse the id  |
| n_train     | int  | number of local training samples (aggregation weight)|

### WELCOME  (server -> client)

| field       | type | notes            |
|-------------|------|------------------|
| type        | str  | "WELCOME"        |
| proto       | int  | protocol version |
| total_rounds| int  | rounds in the run|

### ROUND  (server -> client; start of every round)

| field  | type | notes                                                       |
|--------|------|-------------------------------------------------------------|
| type   | str  | "ROUND"                                                     |
| round  | int  | 1-based round number                                        |
| weights| dict | current global weights: name -> float32 ndarray             |
| config | dict | training settings: local_epochs, batch_size, lr, optimizer, and `clip_norm` when clipped updates are requested |

### UPDATE  (client -> server; answer to a ROUND)

| field     | type | notes                                                        |
|-----------|------|--------------------------------------------------------------|
| type      | str  | "UPDATE"                                                     |
| round     | int  | must equal the round being answered                          |
| client_id | str  | same id as in HELLO                                          |
| n_train   | int  | local sample count                                           |
| delta     | dict | post-training weights minus the weights received in ROUND; same keys and shapes as `weights` |
| aux       | dict | free-form extras; reference fields: train_loss, local_acc, delta_norm, clipped, train_seconds |

### FINISHED  (server -> client)

Sent once after the last round. The client should exit cleanly.

## Round semantics

1. The server broadcasts ROUND `r` to every connected client. The clients to
   which the send succeeded form the "expected" set for the round.
2. Each expected client trains on its local shard and sends exactly one UPDATE `r`.
3. The server waits until every expected update arrived or `round_timeout_s`
   passes. A client that did not answer is logged and skipped for that round;
   the round still completes. The server never blocks forever on one client.
4. The server aggregates, updates the global model, evaluates, next round.

## Aggregation (FedAvg)

Weighted mean of the received deltas:

```
delta_avg = sum over received k of (n_k / sum(n)) * delta_k
global    = global + delta_avg
```

Clients never see each other's data or updates; only the server aggregates.

## Clipping (when `clip_norm` is configured)

When `config.clip_norm` is present in ROUND, the client must scale its delta so
its total L2 norm is at most `clip_norm` before sending
(`scale = clip_norm / ||delta||` when the norm exceeds it). Report the pre-clip
norm and whether clipping happened in `aux.delta_norm` / `aux.clipped`.

## Disconnection and rejoin

- A client that misses `round_timeout_s` is skipped for that round but stays
  connected; a client whose connection dies is marked OFFLINE on the next
  failed read/write.
- The server keeps running with the remaining clients and keeps accepting new
  connections at any time, including between and during rounds.
- On any network error the client closes the socket, waits `retry_delay_s`,
  reconnects and sends HELLO again. The server replies WELCOME; the client then
  waits for the next ROUND, which carries the full current global weights.
  Nothing has to be reconciled: late updates for old rounds are ignored by the
  server, and a reconnected client simply trains from the latest model.
- Identity is `client_id`, never the IP address: a disabled/re-enabled network
  adapter may come back with a different IP.

## Versioning

`proto: 1`. Increment it on any breaking change; the server logs mismatches.

## Conformance testing

`tools/protocol_selftest.py` starts a one-round test server (no training) that
verifies any client implementation: HELLO fields, exactly one UPDATE for the
right round, delta keys/shapes matching the server weights, `n_train > 0`.
Use it to validate a custom client before joining a real run:

```
python tools/protocol_selftest.py --mode all      # reference client self-check
python tools/protocol_selftest.py --mode server   # then point YOUR client at 127.0.0.1:5001
```
