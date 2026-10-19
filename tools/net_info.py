"""Print the addresses teammates can use to reach this machine (run it on the
SERVER machine before telling the group where to connect).

    python tools/net_info.py

Same network (everyone on the same Wi-Fi / LAN):
  share the IPv4 address printed below; it usually starts with 192.168. or 10.

Different networks (each of you at home):
  install Tailscale (or a similar mesh VPN) on every machine, sign in and join
  one shared network, then share the 100.x.y.z address printed below. The
  server keeps working unchanged; only the address in client/config.json
  ("server_host") differs.

Either way the chosen address goes into "server_host" in every client config,
and the port stays the one in server/config.json (default 5000).
"""
from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path


def lan_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("192.0.2.1", 9))  # no packets are actually sent
            return sock.getsockname()[0]
        finally:
            sock.close()
    except OSError:
        return None


def known_ipv4():
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        return sorted({info[4][0] for info in infos})
    except OSError:
        return []


TAILSCALE_PATHS = (
    r"C:\Program Files\Tailscale\tailscale.exe",
    r"C:\Program Files (x86)\Tailscale\tailscale.exe",
)


def tailscale_exe():
    exe = shutil.which("tailscale")
    if exe:
        return exe
    for candidate in TAILSCALE_PATHS:
        if Path(candidate).exists():
            return candidate
    return None


def tailscale_ip():
    exe = tailscale_exe()
    if not exe:
        return None
    try:
        result = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None


def main() -> None:
    print(f"hostname: {socket.gethostname()}")
    print()
    print("Same network (LAN / Wi-Fi) - share this address:")
    primary = lan_ip()
    if primary:
        print(f"  {primary}")
    for extra in known_ipv4():
        if extra != primary and not extra.startswith("100."):
            print(f"  (this machine also answers on {extra})")
    print()
    print("Different networks (Tailscale) - share this address:")
    ts_ip = tailscale_ip()
    tailnet_ips = [ip for ip in known_ipv4() if ip.startswith("100.")]
    if ts_ip:
        print(f"  {ts_ip}")
    elif tailnet_ips:
        print(f"  {tailnet_ips[0]}   (Tailscale interface is up)")
    else:
        print("  Tailscale not found here. Install it on every machine, join the")
        print("  same network, then run this tool again and share the 100.x address.")
    for extra in tailnet_ips:
        if extra != ts_ip:
            print(f"  (also on the tailnet: {extra})")
    print()
    print("Details and the full checklist: TEAM.md")


if __name__ == "__main__":
    main()
