"""Wire protocol v1: framing + encryption + message helpers.

Every message is one frame:  4-byte big-endian length + Fernet(zlib(pickle(dict))).
The full contract is documented in PROTOCOL.md; keep the two in sync.

Threading note: a Connection is used by one sender thread and one receiver
thread at most (server: main thread sends, per-client reader thread receives).
"""
from __future__ import annotations

import pickle
import socket
import struct
import zlib

from cryptography.fernet import Fernet

PROTO_VERSION = 1
MAX_MESSAGE_BYTES = 200 * 1024 * 1024


def load_key(path: str) -> bytes:
    """Read the shared Fernet key (same file on every machine)."""
    with open(path, "rb") as handle:
        return handle.read().strip()


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
        blob = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        blob = zlib.compress(blob, level=1)
        if self.fernet is not None:
            blob = self.fernet.encrypt(blob)
        if len(blob) > MAX_MESSAGE_BYTES:
            raise ValueError(f"message too large: {len(blob)} bytes")
        self.sock.sendall(struct.pack(">I", len(blob)) + blob)

    def recv(self) -> dict:
        (size,) = struct.unpack(">I", self._read_exact(4))
        if size > MAX_MESSAGE_BYTES:
            raise ValueError(f"frame too large: {size} bytes")
        blob = self._read_exact(size)
        if self.fernet is not None:
            blob = self.fernet.decrypt(blob)
        return pickle.loads(zlib.decompress(blob))

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
