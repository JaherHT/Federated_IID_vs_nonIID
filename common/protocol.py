"""Wire protocol v2: framing, safe serialization, encryption.

Every message is one frame:

    4 bytes   payload length N (unsigned, big-endian)
    N bytes   payload = Fernet( zlib( JSON ) )

Serialization rules (full spec in PROTOCOL.md):
  * JSON-safe values only: dict, list, str, int, float, bool, null
  * numpy float32 arrays are carried as
        {"__ndarray__": {"dtype": "float32", "shape": [...], "b64": "..."}}
  * nothing else is accepted on decode

Security note: payloads are plain data. There is no pickle, no eval, no
import, no file paths, no commands. A connected client can only ever deliver
numbers, strings and float32 arrays; the decode side rejects everything else.
Non-finite numbers are rejected on both encode and decode.

Threading note: a Connection is used by one sender thread and one receiver
thread at most (server: main thread sends, per-client reader thread receives).
"""
from __future__ import annotations

import base64
import json
import socket
import struct
import zlib

import numpy as np
from cryptography.fernet import Fernet

PROTO_VERSION = 2
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024


class ProtocolError(ValueError):
    """Message breaks the protocol rules; the connection should be dropped."""


def load_key(path: str) -> bytes:
    """Read the shared Fernet key (same file on every machine)."""
    with open(path, "rb") as handle:
        return handle.read().strip()


def _reject_constant(name: str):
    raise ProtocolError(f"non-finite number rejected: {name}")


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        if obj.dtype != np.float32:
            raise ProtocolError(f"only float32 arrays are sent (got {obj.dtype})")
        return {
            "__ndarray__": {
                "dtype": "float32",
                "shape": [int(dim) for dim in obj.shape],
                "b64": base64.b64encode(obj.tobytes(order="C")).decode("ascii"),
            }
        }
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ProtocolError("non-finite number rejected")
        return obj
    raise ProtocolError(f"unsupported type in message: {type(obj).__name__}")


def _from_jsonable(obj):
    if isinstance(obj, dict):
        if set(obj.keys()) == {"__ndarray__"}:
            meta = obj["__ndarray__"]
            if not isinstance(meta, dict) or meta.get("dtype") != "float32":
                raise ProtocolError("only float32 arrays are accepted")
            shape = meta.get("shape")
            if not isinstance(shape, list) or not shape or not all(
                isinstance(dim, int) and 0 <= dim <= 10_000_000 for dim in shape
            ):
                raise ProtocolError("bad array shape")
            raw = base64.b64decode(meta.get("b64", ""), validate=True)
            flat = np.frombuffer(raw, dtype=np.float32)
            size = 1
            for dim in shape:
                size *= dim
            if size != flat.size:
                raise ProtocolError("array size does not match its shape")
            return flat.reshape(tuple(shape)).copy()
        return {str(key): _from_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(value) for value in obj]
    return obj


def encode_message(message: dict) -> bytes:
    text = json.dumps(_to_jsonable(message), allow_nan=False, separators=(",", ":"))
    return zlib.compress(text.encode("utf-8"), level=1)


def decode_message(blob: bytes) -> dict:
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(blob, MAX_DECOMPRESSED_BYTES)
    if decompressor.unconsumed_tail:
        raise ProtocolError("payload too large after decompression")
    text = raw.decode("utf-8")
    obj = json.loads(text, parse_constant=_reject_constant)
    if not isinstance(obj, dict):
        raise ProtocolError("top-level message must be a dict")
    return _from_jsonable(obj)


class Connection:
    """Framed, encrypted message channel over one TCP socket."""

    def __init__(self, sock: socket.socket, fernet: Fernet | None):
        self.sock = sock
        self.fernet = fernet

    # -- framing helpers ----------------------------------------------------
    def _read_exact(self, n: int) -> bytes:
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise ConnectionError("connection closed by peer")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    # -- public API ----------------------------------------------------------
    def send(self, message: dict) -> None:
        blob = encode_message(message)
        if self.fernet is not None:
            blob = self.fernet.encrypt(blob)
        if len(blob) > MAX_MESSAGE_BYTES:
            raise ProtocolError(f"message too large: {len(blob)} bytes")
        self.sock.sendall(struct.pack(">I", len(blob)) + blob)

    def recv(self) -> dict:
        (size,) = struct.unpack(">I", self._read_exact(4))
        if size > MAX_MESSAGE_BYTES:
            raise ProtocolError(f"frame too large: {size} bytes")
        blob = self._read_exact(size)
        if self.fernet is not None:
            blob = self.fernet.decrypt(blob)
        return decode_message(blob)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def from_socket(sock: socket.socket, key: bytes | None) -> Connection:
    return Connection(sock, Fernet(key) if key is not None else None)


def connect(host: str, port: int, key: bytes | None, timeout_s: float) -> Connection:
    """Dial the server (connection phase is bounded by timeout_s)."""
    sock = socket.create_connection((host, port), timeout=timeout_s)
    sock.settimeout(None)
    return from_socket(sock, key)


def make_listener(host: str, port: int) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(16)
    return listener
