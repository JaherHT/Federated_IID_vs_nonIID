"""Generate the shared Fernet key used by the wire protocol.

    python tools/gen_key.py            # writes secret.key in the repo root
    python tools/gen_key.py --force    # overwrite an existing key

The SAME file must be present on the server and on every client machine
(copy it out of band: USB stick, private chat). Never commit it to git.
If a teammate's key differs, their client cannot talk to the server: the
first message fails with `InvalidToken`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cryptography.fernet import Fernet


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate the shared protocol key")
    parser.add_argument("--out", default="secret.key")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.out)
    if path.exists() and not args.force:
        print(f"{path} already exists; use --force to overwrite (that breaks every machine still using the old key)")
        return
    path.write_bytes(Fernet.generate_key())
    print(f"wrote {path}")
    print("copy this exact file to every machine (server + all clients) out of band.")


if __name__ == "__main__":
    main()
