#!/usr/bin/env python3
"""Plot velocity magnitude and vorticity fields written by the solver.

Usage: plot_fields.py OUTPUT_DIR [--png PATH]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--png", type=Path, default=None,
                        help="output image (default: OUTPUT_DIR/fields.png)")
    args = parser.parse_args()

    d = args.output_dir
    ux = np.load(d / "ux.npy")
    uy = np.load(d / "uy.npy")
    vort = np.load(d / "vorticity.npy")
    solid = np.load(d / "solid.npy").astype(bool)

    speed = np.ma.masked_where(solid, np.hypot(ux, uy))
    vort = np.ma.masked_where(solid, vort)
    vmax = float(np.abs(vort).max()) or 1.0

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    im0 = axes[0].imshow(speed, origin="lower", cmap="viridis")
    axes[0].set_title("|u| (lattice units)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(vort, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("vorticity")
    fig.colorbar(im1, ax=axes[1])
    for ax in axes:
        ax.set_ylabel("y")
    axes[1].set_xlabel("x")

    fig.tight_layout()
    png = args.png or d / "fields.png"
    fig.savefig(png, dpi=120)
    print(f"-> {png}")


if __name__ == "__main__":
    main()
