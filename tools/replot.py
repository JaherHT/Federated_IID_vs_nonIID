"""Rebuild metrics.png for existing runs from their CSVs.

The server draws the figure when a run ends (or is stopped, or crashes). This
tool redraws it for runs already on disk, using the same code path, so an old
run gets the current look without re-running the federation.

    python tools/replot.py                       # every run under runs/ with a metrics.csv
    python tools/replot.py runs/20261020_211344 # one run
    python tools/replot.py runs/20261020_211344 --label "non-IID shards"
    python tools/replot.py --all --dpi 220       # larger PNGs

A run interrupted with Ctrl-C has no summary of its own; the end reason is read
from server.log, so the figure still marks where the server stopped.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.figures import render_run_figure  # noqa: E402


def read_csv(path: Path) -> list:
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def partition_of_run(run_dir: Path, override=None) -> str:
    """Partition label from the run artifacts, or --partition when they predate it."""
    if override:
        return str(override)
    found = set()
    for name in ("run_config.json", "summary.json"):
        parts = read_json(run_dir / name).get("client_partitions")
        if isinstance(parts, dict):
            found |= {str(v) for v in parts.values() if v and str(v) != "unknown"}
    if len(found) == 1:
        return found.pop()
    if len(found) > 1:
        return "mixed"
    meta = read_json(run_dir / "figure_meta.json").get("partition")
    return str(meta) if meta else ""


def end_reason(run_dir: Path) -> str:
    summary = run_dir / "summary.json"
    if summary.exists():
        try:
            ended = json.loads(summary.read_text(encoding="utf-8-sig")).get("ended")
            if isinstance(ended, dict) and ended.get("reason"):
                return str(ended["reason"])
        except (OSError, ValueError):
            pass
    log = run_dir / "server.log"
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        if "RUN FINISHED" in text:
            return "completed"
        if "interrupted by user" in text:
            return "interrupted"
        if "run aborted by an uncaught error" in text or "Traceback (most recent call last)" in text:
            return "error"
    return "unknown"


def redraw(run_dir: Path, out_dir: Path, label=None, partition=None, dpi: int = 170) -> str:
    metrics_path = run_dir / "metrics.csv"
    rows = read_csv(metrics_path)
    if not rows:
        return f"{run_dir.name}: skipped (no metrics.csv)"
    client_rows = read_csv(run_dir / "client_metrics.csv")

    config = {}
    strategy = "fedavg"
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
            config = payload.get("config") or {}
            strategy = str(payload.get("strategy") or config.get("strategy") or "fedavg")
        except (OSError, ValueError):
            pass

    expected_ids = [str(c) for c in (config.get("expected_client_ids") or [])]
    reason = end_reason(run_dir)
    path = render_run_figure(
        out_dir, rows, client_rows,
        strategy=strategy,
        expected_client_ids=expected_ids,
        expected_clients=config.get("expected_clients"),
        round_timeout_s=config.get("round_timeout_s"),
        end_reason=reason,
        end_round=max(int(float(r["round"])) for r in rows),
        label=label,
        partition=partition_of_run(run_dir, partition),
        dpi=dpi,
    )
    last = rows[-1]
    return (f"{run_dir.name}: {path.name} | {strategy} | {partition_of_run(run_dir, partition) or 'partition n/a'} | "
            f"{len(rows) - 1} rounds | "
            f"final acc {float(last['test_acc']):.3f} | {reason}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Redraw metrics.png from run CSVs")
    parser.add_argument("runs", nargs="*", help="run directories (default: all under runs/)")
    parser.add_argument("--all", action="store_true", help="every run under runs/ (default)")
    parser.add_argument("--label", default=None, help="extra words for the figure title")
    parser.add_argument("--partition", choices=["iid", "non-iid", "mixed"], default=None,
                        help="partition label when the run predates client_partitions")
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--out", default=None, help="write elsewhere (default: into each run dir)")
    args = parser.parse_args(argv)

    if args.runs:
        targets = [Path(item) for item in args.runs]
        targets = [t if t.is_absolute() else (REPO_ROOT / t) for t in targets]
    else:
        targets = sorted(p for p in (REPO_ROOT / "runs").iterdir() if p.is_dir())

    failures = 0
    for run_dir in targets:
        try:
            out_dir = Path(args.out) / run_dir.name if args.out else run_dir
            print(redraw(run_dir, out_dir, label=args.label, partition=args.partition, dpi=args.dpi))
        except Exception as exc:  # keep going: one broken run must not stop the batch
            failures += 1
            print(f"{run_dir.name}: FAILED ({type(exc).__name__}: {exc})")

    if failures:
        print(f"\n{failures} run(s) could not be redrawn")
        sys.exit(1)


if __name__ == "__main__":
    main()
