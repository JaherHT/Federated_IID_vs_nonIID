"""Server-side report figure for a federated run.

`render_run_figure` draws the end-of-run panel set from the rows the server
already keeps in memory, or from a finished run's CSVs (`tools/replot.py`).
matplotlib is imported inside the function, so clients and machines without it
never touch it.

Panels, top to bottom:

  1. global test accuracy, rounds that lost a client shaded
  2. global test loss, same shading
  3. per-client local accuracy, line broken at every missed round
  4. round duration, bars coloured by whether everyone asked answered
  5. client participation matrix: answered / missed / dropped / rejected / no data

Everything is read from the rows: no run state is guessed. A client that never
connected shows up as "no data" for every round; a client that missed a round
shows a gap plus a red x in panel 3 and a coloured cell in panel 5.
"""
from __future__ import annotations

import math
from pathlib import Path

STATUS_CODE = {"ok": 1, "miss": 2, "dead": 3, "bad-update": 4}
STATUS_COLORS = ["#E6E9ED", "#3F8E7A", "#E1A33F", "#B4232A", "#8E5BA6"]
STATUS_NAMES = ["no data", "answered", "missed (timeout)", "dropped", "rejected"]
PARTITION_TEXT = {"iid": "IID shards", "non-iid": "non-IID shards",
                  "mixed": "mixed IID / non-IID shards"}
CLIENT_COLORS = ["#1D4E89", "#2C7A7B", "#B7791F", "#7B4B94", "#4A7C59"]
INK = "#2D3748"
MUTED = "#4A5568"
GRID = "#CBD5E0"
ACCENT = "#1D4E89"
LOSS = "#A4343A"
ALERT = "#B4232A"
SHADE = "#E7A6A1"
TIMEOUT = "#5A6B7B"


def _num(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _partition_text(value) -> str:
    """Turn the reported partition ("iid"/"non-iid"/"mixed") into title words."""
    if not value:
        return ""
    key = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    return PARTITION_TEXT.get(key, str(value).strip())


def _int(value):
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _prepare(metrics_rows, client_rows, expected_client_ids, expected_clients):
    rows = []
    for row in metrics_rows or []:
        r = _int(row.get("round"))
        if r is None:
            continue
        rows.append({
            "round": r,
            "acc": _num(row.get("test_acc")),
            "loss": _num(row.get("test_loss")),
            "wall": _num(row.get("wall_s")),
            "alive": _int(row.get("clients_alive")),
            "ok": _int(row.get("updates_ok")),
        })
    rows.sort(key=lambda item: item["round"])
    last_round = max((item["round"] for item in rows), default=0)

    status: dict = {}
    client_acc: dict = {}
    for row in client_rows or []:
        r = _int(row.get("round"))
        cid = str(row.get("client_id") or "").strip()
        if r is None or not cid:
            continue
        state = str(row.get("status") or "").strip()
        status[(r, cid)] = state
        if state == "ok":
            value = _num(row.get("local_acc"))
            if value is not None:
                client_acc.setdefault(cid, {})[r] = value

    observed = sorted({cid for (_, cid) in status})
    client_ids = [str(c) for c in (expected_client_ids or [])]
    client_ids += [cid for cid in observed if cid not in client_ids]
    if not client_ids:
        client_ids = [f"client {i + 1}" for i in range(int(expected_clients or 0))]

    events: dict = {}
    for (r, cid), state in status.items():
        if state != "ok" and 1 <= r <= last_round:
            events.setdefault(r, {})[cid] = state

    return {
        "rows": rows,
        "last_round": last_round,
        "status": status,
        "client_acc": client_acc,
        "client_ids": client_ids,
        "events": events,
        "n_events": sum(len(v) for v in events.values()),
        "n_misses": sum(1 for s in status.values() if s == "miss"),
        "n_drops": sum(1 for s in status.values() if s == "dead"),
    }


def build_run_figure(metrics_rows, client_rows, *, strategy="fedavg",
                     expected_client_ids=None, expected_clients=None,
                     round_timeout_s=None, end_reason="completed",
                     end_round=None, label=None, partition=None, dpi=170):
    """Draw the figure and return it. The caller owns (and closes) the Figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter, MaxNLocator
    except Exception as exc:  # pragma: no cover - machine dependent
        raise RuntimeError(f"matplotlib is not installed ({exc})") from exc

    data = _prepare(metrics_rows, client_rows, expected_client_ids, expected_clients)
    rows = data["rows"]
    if not rows:
        raise ValueError("no metrics rows to plot")

    last_round = data["last_round"]
    acc_points = [(item["round"], item["acc"]) for item in rows if item["acc"] is not None]
    loss_points = [(item["round"], item["loss"]) for item in rows if item["loss"] is not None]
    timed = [item for item in rows if item["round"] >= 1 and item["wall"] is not None]
    events = data["events"]
    client_ids = data["client_ids"]
    client_acc = data["client_acc"]
    finished = str(end_reason or "completed") == "completed"
    stop_round = int(end_round if end_round is not None else last_round)

    def style(ax, grid_axis="y"):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
        ax.grid(axis=grid_axis, color=GRID, alpha=0.55, linewidth=0.7,
                linestyle=(0, (3, 3)))
        ax.set_axisbelow(True)

    def int_ticks(ax, start=1):
        if 0 < last_round <= 20:
            ax.set_xticks(range(start, last_round + 1))
        else:
            step = 5 if last_round <= 60 else (10 if last_round <= 150 else 20)
            first = start if start == 0 else step
            ax.set_xticks(range(first, last_round + 1, step))

    def fit_ticks(ax, axis="y"):
        """Drop tick labels that fall outside the current view (e.g. 120%)."""
        limits = ax.get_ylim() if axis == "y" else ax.get_xlim()
        low, high = limits
        current = ax.get_yticks() if axis == "y" else ax.get_xticks()
        keep = [tick for tick in current if low - 1e-9 <= tick <= high + 1e-9]
        if keep:
            if axis == "y":
                ax.set_yticks(keep)
            else:
                ax.set_xticks(keep)

    def pct_axis(ax):
        """Percent ticks on round steps, so labels never read 82%, 85%, 88%."""
        low, high = ax.get_ylim()
        span = max(high - low, 1e-9)
        if span < 0.02:  # tiny range: keep the default ticks, just more precision
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.1f}%"))
            return
        step, ticks = 0.5, []
        for candidate in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5):
            start = math.ceil(low / candidate - 1e-9) * candidate
            found, value = [], start
            while value <= high + 1e-9:
                found.append(round(value, 6))
                value += candidate
            if len(found) <= 8:
                step, ticks = candidate, found
                if len(found) >= 3:
                    break
        if len(ticks) >= 2:
            ax.set_yticks(ticks)
        decimals = 0 if step >= 0.01 - 1e-9 else 1
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%"))

    def shade(ax):
        for r in sorted(events):
            ax.axvspan(r - 0.42, r + 0.42, color=SHADE, alpha=0.32, linewidth=0, zorder=0)

    fig = plt.figure(figsize=(13.4, 9.2), dpi=dpi, layout="constrained")
    grid = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 0.95, 0.40])
    ax_acc = fig.add_subplot(grid[0, 0])
    ax_loss = fig.add_subplot(grid[0, 1])
    ax_client = fig.add_subplot(grid[1, 0])
    ax_time = fig.add_subplot(grid[1, 1])
    bottom = grid[2, :].subgridspec(1, 2, width_ratios=[0.87, 0.13])
    ax_part = fig.add_subplot(bottom[0, 0])
    ax_key = fig.add_subplot(bottom[0, 1])  # holds the colour key, keeps it off the edge
    ax_key.set_axis_off()
    ax_key.set_xticks([])
    ax_key.set_yticks([])
    for ax in (ax_acc, ax_loss, ax_client, ax_time, ax_part):
        style(ax)

    # ------------------------------------------------------------ accuracy
    if acc_points:
        ax_acc.plot([p[0] for p in acc_points], [p[1] for p in acc_points],
                    color=ACCENT, linewidth=2.2, marker="o", markersize=5.2,
                    markerfacecolor="white", markeredgewidth=1.6, zorder=3)
        best_round, best_acc = max(acc_points, key=lambda p: p[1])
        ax_acc.plot([best_round], [best_acc], marker="*", markersize=13,
                    color="#D69E2E", markeredgecolor="white", markeredgewidth=0.8,
                    linestyle="none", zorder=4)
        ax_acc.annotate(f"best {best_acc * 100:.1f}%  (r{best_round})",
                        xy=(best_round, best_acc), xytext=(-10, 4),
                        textcoords="offset points", ha="right", fontsize=8.5,
                        color=INK, fontweight="bold")
        low_acc = min(p[1] for p in acc_points)
        span = max(best_acc - low_acc, 0.05)
        top = min(1.0, best_acc + 0.18 * span)
        if 1.0 - top <= 0.06:  # snap so the 100% line stays labelled
            top = 1.0
        ax_acc.set_ylim(max(0.0, low_acc - 0.12 * span), top)
    shade(ax_acc)
    if events:
        ax_acc.legend(handles=[Patch(facecolor=SHADE, alpha=0.55, edgecolor="none",
                                     label="client missed this round")],
                      loc="lower right", fontsize=8, frameon=False)
    ax_acc.set_title("Global test accuracy", loc="left", fontsize=11, color=INK,
                     fontweight="bold", pad=8)
    ax_acc.set_xlabel("round", fontsize=9, color=MUTED)
    ax_acc.set_ylabel("accuracy", fontsize=9, color=MUTED)
    int_ticks(ax_acc, start=0)
    ax_acc.set_xlim(-0.6, last_round + 0.8)
    pct_axis(ax_acc)

    # ---------------------------------------------------------------- loss
    if loss_points:
        ax_loss.plot([p[0] for p in loss_points], [p[1] for p in loss_points],
                     color=LOSS, linewidth=2.0, marker="s", markersize=4.6,
                     markerfacecolor="white", markeredgewidth=1.4, zorder=3)
        low_round, low_loss = min(loss_points, key=lambda p: p[1])
        ax_loss.annotate(f"min {low_loss:.3f}  (r{low_round})",
                         xy=(low_round, low_loss), xytext=(10, 12),
                         textcoords="offset points", fontsize=8.5, color=INK,
                         fontweight="bold",
                         arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    shade(ax_loss)
    ax_loss.set_title("Global test loss", loc="left", fontsize=11, color=INK,
                      fontweight="bold", pad=8)
    ax_loss.set_xlabel("round", fontsize=9, color=MUTED)
    ax_loss.set_ylabel("cross-entropy", fontsize=9, color=MUTED)
    int_ticks(ax_loss, start=0)
    ax_loss.set_xlim(-0.6, last_round + 0.8)
    fit_ticks(ax_loss, "y")

    # ------------------------------------------------------------- clients
    plotted = 0
    miss_points = []  # (client, round, y or None): None = dropped before its first update
    for index, cid in enumerate(client_ids):
        series = client_acc.get(cid) or {}
        if not series:
            continue
        index_rounds = list(range(1, last_round + 1))
        values = [series.get(r) for r in index_rounds]
        color = CLIENT_COLORS[index % len(CLIENT_COLORS)]
        ax_client.plot(index_rounds, values, color=color, linewidth=1.8,
                       marker="o", markersize=4.2, label=cid, zorder=3)
        last_known = None
        for r in index_rounds:
            if series.get(r) is not None:
                last_known = series[r]
            elif (r, cid) in data["status"]:
                miss_points.append((cid, r, last_known))
        plotted += 1
    if plotted:
        handles, labels = ax_client.get_legend_handles_labels()
        if data["n_events"]:
            handles.append(Line2D([], [], marker="x", color=ALERT, linestyle="none",
                                  markersize=7.5, markeredgewidth=1.9))
            labels.append("missed / dropped round")
        ax_client.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.005),
                         fontsize=8, ncol=len(labels), frameon=False,
                         handlelength=1.4, handletextpad=0.5, columnspacing=1.3)
        values = [v for series in client_acc.values() for v in series.values()]
        low, high = min(values), max(values)
        pad = max(high - low, 0.02)
        ax_client.set_ylim(max(0.0, low - 0.10 * pad), min(1.02, high + 0.34 * pad))
        floor = ax_client.get_ylim()[0] + 0.03 * (ax_client.get_ylim()[1] - ax_client.get_ylim()[0])
        for cid, r, y in miss_points:
            ax_client.plot([r], [y if y is not None else floor], marker="x",
                           markersize=7.5, markeredgewidth=1.9, color=ALERT,
                           linestyle="none", zorder=4)
            if y is None:  # dropped before its first update: name it, the gap has no line
                ax_client.annotate(f"{cid} dropped", xy=(r, floor),
                                   xytext=(4, 3), textcoords="offset points",
                                   fontsize=7.5, color=ALERT, ha="left", va="bottom",
                                   bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                                             edgecolor="none", alpha=0.8))
    else:
        ax_client.text(0.5, 0.5, "no per-client local metrics in this run",
                       transform=ax_client.transAxes, ha="center", va="center",
                       fontsize=9, color=MUTED)
    int_ticks(ax_client)
    ax_client.set_xlim(-0.6, last_round + 0.8)
    pct_axis(ax_client)
    ax_client.set_title("Client local accuracy (in-sample on each shard)",
                        loc="left", fontsize=11, color=INK, fontweight="bold", pad=8)
    ax_client.set_xlabel("round", fontsize=9, color=MUTED)
    ax_client.set_ylabel("accuracy", fontsize=9, color=MUTED)

    # ------------------------------------------------------------ duration
    timeout_note = ""
    if timed:
        xs = [item["round"] for item in timed]
        ys = [item["wall"] for item in timed]
        colors = []
        for item in timed:
            if not events.get(item["round"]):
                colors.append("#3F6C82")
            elif (item["ok"] or 0) > 0:
                colors.append("#D6923E")
            else:
                colors.append(ALERT)
        ax_time.bar(xs, ys, width=0.62, color=colors, edgecolor="white", linewidth=0.6)
        tallest = max(range(len(ys)), key=lambda i: ys[i])
        ax_time.annotate(f"{ys[tallest]:.0f} s", xy=(xs[tallest], ys[tallest]),
                         xytext=(0, 4), textcoords="offset points", ha="center",
                         fontsize=8, color=INK, fontweight="bold")
        top = max(ys) * 1.22
        handles = []
        for color, text_label in (("#3F6C82", "all answered"), ("#D6923E", "client missing"),
                                  (ALERT, "no updates")):
            if color in colors:
                handles.append(Patch(facecolor=color, edgecolor="white", label=text_label))
        if round_timeout_s:
            if round_timeout_s <= max(ys) * 1.35:
                ax_time.axhline(round_timeout_s, color=TIMEOUT, linewidth=1.2,
                                linestyle=(0, (4, 3)), zorder=2)
                handles.append(Line2D([], [], color=TIMEOUT, linewidth=1.2,
                                      linestyle=(0, (4, 3)),
                                      label=f"round timeout ({round_timeout_s:.0f} s)"))
                top = max(top, round_timeout_s * 1.10)
            else:
                timeout_note = f" · timeout {round_timeout_s:.0f} s"
        if handles:
            ax_time.legend(handles=handles, loc="upper left", fontsize=7.8, frameon=False,
                           ncol=len(handles), handlelength=1.1, handletextpad=0.4,
                           columnspacing=1.0)
        ax_time.set_ylim(0, top)
        ax_time.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        mean_complete = [item["wall"] for item in timed if not events.get(item["round"])]
        note = f" — mean {sum(mean_complete) / len(mean_complete):.1f} s" if mean_complete else ""
    else:
        ax_time.text(0.5, 0.5, "no completed round", transform=ax_time.transAxes,
                     ha="center", va="center", fontsize=9, color=MUTED)
        note = ""
    ax_time.set_title(f"Round duration{note}{timeout_note}", loc="left", fontsize=11,
                      color=INK, fontweight="bold", pad=8)
    ax_time.set_xlabel("round", fontsize=9, color=MUTED)
    ax_time.set_ylabel("seconds", fontsize=9, color=MUTED)
    int_ticks(ax_time)
    ax_time.set_xlim(0.4, last_round + 0.6)
    fit_ticks(ax_time, "y")

    # ------------------------------------------------------- participation
    matrix = []
    for cid in client_ids:
        line = []
        for r in range(1, last_round + 1):
            line.append(STATUS_CODE.get(data["status"].get((r, cid), ""), 0))
        matrix.append(line)
    if last_round < 1 or not matrix:
        ax_part.axis("off")
        message = ("no rounds completed" if last_round < 1
                   else "no client participation recorded")
        ax_part.text(0.5, 0.5, message, transform=ax_part.transAxes,
                     ha="center", va="center", fontsize=9, color=MUTED)
    else:
        ax_part.imshow(matrix, aspect="auto", interpolation="nearest",
                       cmap=ListedColormap(STATUS_COLORS), vmin=-0.5, vmax=4.5)
        for row_index, line in enumerate(matrix):
            for column_index, code in enumerate(line):
                if code >= 2:
                    ax_part.text(column_index, row_index, "×", ha="center", va="center",
                                 color="white", fontsize=10, fontweight="bold")
        labels = []
        for row_index, cid in enumerate(client_ids):
            counts = {code: sum(1 for c in matrix[row_index] if c == code) for code in (2, 3, 4)}
            marks = []
            if counts[2]:
                marks.append(f"{counts[2]} missed")
            if counts[3]:
                marks.append(f"{counts[3]} dropped")
            if counts[4]:
                marks.append(f"{counts[4]} rejected")
            labels.append(f"{cid}  ({', '.join(marks)})" if marks else cid)
        ax_part.set_yticks(range(len(client_ids)))
        ax_part.set_yticklabels(labels, fontsize=8.5, color=INK)
        ax_part.set_xticks(range(last_round))
        ax_part.set_xticklabels([str(r) for r in range(1, last_round + 1)], fontsize=8)
        ax_part.set_xticks([x - 0.5 for x in range(1, last_round + 1)], minor=True)
        ax_part.set_yticks([y - 0.5 for y in range(1, len(client_ids))], minor=True)
        ax_part.grid(which="minor", color="white", linewidth=1.4)
        ax_part.grid(which="major", visible=False)
        ax_part.tick_params(which="minor", length=0)
        ax_part.tick_params(axis="both", length=0)
        for side in ("top", "right", "left", "bottom"):
            ax_part.spines[side].set_visible(False)
        answered = sum(1 for line in matrix for code in line if code == 1)
        slots = len(client_ids) * last_round
        tally = f"{answered}/{slots} updates"
        if data["n_misses"]:
            tally += f" · {data['n_misses']} missed"
        if data["n_drops"]:
            tally += f" · {data['n_drops']} dropped"
        ax_part.set_title(f"Client participation per round — {tally}", loc="left",
                          fontsize=11, color=INK, fontweight="bold", pad=6)
        present = [0, 1] + [code for code in (2, 3, 4)
                            if any(code in line for line in matrix)]
        handles = [Patch(facecolor=STATUS_COLORS[i], edgecolor="#B8BFC8", linewidth=0.6,
                         label=STATUS_NAMES[i]) for i in present]
        ax_key.legend(handles=handles, loc="center left", frameon=False, fontsize=8,
                     ncol=1, handlelength=1.1, handleheight=1.1, borderaxespad=0.1)
        ax_part.set_xlabel("round", fontsize=9, color=MUTED)

    # ---------------------------------------------------------- end marker
    if not finished:
        for ax in (ax_acc, ax_loss):
            ax.axvline(stop_round + 0.45, color=ALERT, linewidth=1.3,
                       linestyle=(0, (5, 3)), zorder=2)
        reason = "interrupted" if str(end_reason) == "interrupted" else str(end_reason)
        # bottom-left is empty when the curve starts low and climbs; top-left otherwise
        badge_y, badge_va = ((0.03, "bottom") if acc_points and min(p[1] for p in acc_points) < 0.5
                             else (0.965, "top"))
        ax_acc.text(0.015, badge_y,
                    f"INCOMPLETE — server stopped at round {stop_round} ({reason})",
                    transform=ax_acc.transAxes, ha="left", va=badge_va, fontsize=8.5,
                    color=ALERT, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.32", facecolor="#FDECEA",
                              edgecolor=SHADE, linewidth=0.8))

    # -------------------------------------------------------------- titles
    strategy_name = {"fedavg": "FedAvg", "logits": "logits exchange"}.get(
        str(strategy).lower(), str(strategy))
    pieces = [strategy_name]
    partition_text = _partition_text(partition)
    if partition_text:
        pieces.append(partition_text)
    if label:
        pieces.append(str(label))
    observed = sum(1 for cid in client_ids
                   if any((r, cid) in data["status"] for r in range(1, last_round + 1)))
    expected_n = max(len([c for c in (expected_client_ids or [])]),
                     int(expected_clients or 0))
    if expected_n and 0 < observed < expected_n:
        pieces.append(f"{observed} of {expected_n} clients")
    elif expected_n:
        pieces.append(f"{expected_n} client{'s' if expected_n != 1 else ''}")
    elif observed:
        pieces.append(f"{observed} client{'s' if observed != 1 else ''}")
    pieces.append(f"{last_round} round{'s' if last_round != 1 else ''}")
    if acc_points:
        head = "final test accuracy" if finished else f"accuracy at round {stop_round}"
        pieces.append(f"{head} {acc_points[-1][1] * 100:.1f}%")
    if not finished:
        pieces.append(f"INCOMPLETE (stopped at round {stop_round}, {end_reason})")
    fig.suptitle(" · ".join(pieces), fontsize=12.5, color=INK, fontweight="bold",
                 x=0.012, ha="left")
    return fig


def render_run_figure(out_dir, metrics_rows, client_rows, *, strategy="fedavg",
                      expected_client_ids=None, expected_clients=None,
                      round_timeout_s=None, end_reason="completed",
                      end_round=None, label=None, partition=None, filename="metrics.png",
                      dpi=170):
    """Write `<out_dir>/<filename>` and return its path. Raises if nothing to draw."""
    fig = build_run_figure(
        metrics_rows, client_rows, strategy=strategy,
        expected_client_ids=expected_client_ids, expected_clients=expected_clients,
        round_timeout_s=round_timeout_s, end_reason=end_reason,
        end_round=end_round, label=label, partition=partition, dpi=dpi,
    )
    try:
        import matplotlib.pyplot as plt
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches=None)
        return path
    finally:
        plt.close(fig)
