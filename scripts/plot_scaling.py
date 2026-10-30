#!/usr/bin/env python3
"""Parallel efficiency of v4 against the number of GPUs, from run_scaling.py CSV files.

Only rows measured with one GPU per rank (valid_scaling = 1) are plotted; if a file contains
rows where ranks share a device, the script stops: such numbers say nothing about scaling.

Usage: plot_scaling.py results/v4_strong_scaling_*.csv [results/v4_weak_scaling_*.csv]
           [--png docs/scaling.png]
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"strong": "#2a78d6", "weak": "#eb6834"}  # reference palette, slots 1 and 2
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--png", type=Path, default=Path("docs/scaling.png"))
    args = ap.parse_args()

    series = {}
    for path in args.csv:
        rows = list(csv.DictReader(open(path)))
        shared = [r for r in rows if r["valid_scaling"] != "1"]
        if shared:
            sys.exit(f"{path}: {len(shared)} row(s) measured with ranks sharing a device "
                     f"(or on the CPU engine); not a scaling measurement, nothing plotted.")
        for r in rows:
            series.setdefault(r["mode"], []).append((int(r["ranks"]), float(r["efficiency"])))

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    top = max(p for s in series.values() for p, _ in s)
    ax.plot([1, top], [1, 1], color=TEXT_2, linestyle=(0, (5, 4)), linewidth=1.5, label="ideal")
    for mode, pts in sorted(series.items()):
        pts.sort()
        ax.plot([p for p, _ in pts], [e for _, e in pts], marker="o", markersize=8, linewidth=2,
                color=COLORS.get(mode, TEXT), label=f"{mode} scaling")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({p for s in series.values() for p, _ in s}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("GPUs (one MPI rank each)", color=TEXT_2)
    ax.set_ylabel("parallel efficiency", color=TEXT_2)
    ax.set_title("v4 parallel efficiency", loc="left", color=TEXT)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=130, facecolor=SURFACE)
    print(f"-> {args.png}")


if __name__ == "__main__":
    main()
