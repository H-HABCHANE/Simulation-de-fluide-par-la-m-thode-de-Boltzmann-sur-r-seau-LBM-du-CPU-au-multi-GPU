#!/usr/bin/env python3
"""MLUPS of each version against the memory-bandwidth ceiling of its machine.

Two panels, one per machine (CPU and GPU are different machines with different scales; they
are not drawn on a shared or dual axis). Each bar is one version at the same grid size; the
dashed line is the measured ceiling, bandwidth / (2 * 9 * sizeof(real)), and each bar is
labelled with its MLUPS and its percentage of that ceiling. Every value is read from
results/*.csv.

Usage: plot_mlups.py [--size 2048] [--png docs/mlups_vs_ceiling.png]
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
BYTES_PER_CELL = 2 * 9 * 4  # float

# Reference palette (light mode): slot 1 for the CPU panel, slot 2 for the GPU panel.
CPU_COLOR, GPU_COLOR = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def rows(name):
    path = RESULTS / name
    return list(csv.DictReader(open(path))) if path.exists() else []


def mlups(name, size, **match):
    """Mean MLUPS of the row of `name` at size x size matching the given columns."""
    for r in rows(name):
        if int(r["nx"]) == size and int(r["ny"]) == size and all(r.get(k) == v for k, v in match.items()):
            return float(r["mlups_mean"]), float(r["mlups_std"])
    return None


def ceiling(name):
    best = max(max(float(r["copy_MBps"]), float(r.get("memcpy_MBps") or 0)) for r in rows(name))
    return best * 1e6 / BYTES_PER_CELL / 1e6


def panel(ax, title, bars, ceil, color, ceil_label):
    labels = [b[0] for b in bars]
    values = [b[1][0] for b in bars]
    errors = [b[1][1] for b in bars]
    x = range(len(bars))
    ax.bar(x, values, width=0.6, color=color, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, values, yerr=errors, fmt="none", ecolor=TEXT, capsize=3, zorder=6)
    ax.axhline(ceil, color=TEXT_2, linestyle=(0, (5, 4)), linewidth=1.5, zorder=4)
    top = max(ceil, max(values)) * 1.08
    # Values inside the top of tall bars (white ink), above short ones: no label ever sits
    # on the ceiling line. The ceiling value is given under the panel title.
    for i, v in enumerate(values):
        text = f"{v:,.0f}\n{100 * v / ceil:.0f} %"
        if v > 0.3 * top:
            ax.text(i, v - top * 0.02, text, ha="center", va="top", color="#ffffff", fontsize=9,
                    fontweight="bold", zorder=5)
        else:
            ax.text(i, v + top * 0.02, text, ha="center", va="bottom", color=TEXT, fontsize=9,
                    zorder=5)
    ax.set_xlim(-0.6, max(len(bars), 3) - 0.4)  # same bar width whatever the bar count
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, color=TEXT)
    ax.set_ylim(0, top)
    ax.set_title(f"{title}\n", loc="left", fontsize=11, color=TEXT)
    ax.text(0, 1.02, f"dashed line: {ceil_label}, {ceil:,.0f} MLUPS", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color=TEXT_2)
    ax.set_ylabel("MLUPS (higher is better)", color=TEXT_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--png", type=Path, default=Path("docs/mlups_vs_ceiling.png"))
    args = ap.parse_args()
    n = args.size

    cpu = [("v0\nsequential", mlups("v0_i5-10310U.csv", n)),
           ("v1_naive\nOpenMP x4", mlups("v1_naive_i5-10310U.csv", n, threads="4")),
           ("v1\nOpenMP x4, SoA+SIMD", mlups("v1_i5-10310U.csv", n, threads="4"))]
    cpu = [b for b in cpu if b[1]]

    steps = "v3_steps_Tesla-T4.csv"
    if rows(steps):  # same-session comparison of v2 and the v3 variants
        gpu = [("v2\nnaive (AoS, push)", mlups(steps, n, version="v2"))]
        gpu += [(f"v3\n{v.replace('_', ' ')}", mlups(steps, n, version="v3", variant=v))
                for v in ("soa_push", "soa_pull", "soa_pull_opt")]
    else:
        gpu = [("v2\nnaive (AoS, push)", mlups("v2_Tesla-T4.csv", n))]
    gpu = [b for b in gpu if b[1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [max(len(cpu), 3), max(len(gpu), 3)]})
    fig.patch.set_facecolor(SURFACE)
    panel(ax1, "CPU — Intel i5-10310U (4 cores)", cpu, ceiling("bandwidth_i5-10310U.csv"),
          CPU_COLOR, "STREAM ceiling")
    panel(ax2, "GPU — NVIDIA Tesla T4", gpu, ceiling("gpu_bandwidth_Tesla-T4.csv"), GPU_COLOR,
          "measured ceiling")
    fig.suptitle(f"D2Q9 LBM, float, {n}x{n} grid: MLUPS vs memory-bandwidth ceiling",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.text(0.01, 0.01, "Ceiling = measured bandwidth / 72 bytes per cell update. "
             "Error bars: standard deviation over 5 runs. Data: results/*.csv.",
             fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=130, facecolor=SURFACE)
    print(f"-> {args.png}")


if __name__ == "__main__":
    main()
