#!/usr/bin/env python3
"""Compare the simulated Poiseuille profile with the analytical parabola.

Usage: plot_poiseuille.py OUTPUT_DIR [--png PATH]
OUTPUT_DIR is the directory written by lbm_v0 (ux.npy, meta.json).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--png", type=Path, default=Path("docs/poiseuille_profile.png"))
    args = parser.parse_args()

    meta = json.loads((args.output_dir / "meta.json").read_text())
    ux = np.load(args.output_dir / "ux.npy").astype(np.float64)

    ny, tau, force = meta["ny"], meta["tau"], meta["force_x"]
    H = ny - 2
    nu = (tau - 0.5) / 3.0

    # Fluid rows y = 1 .. ny-2; walls (halfway bounce-back) at y = 0.5 and y = ny - 1.5.
    profile = ux[1:-1, :].mean(axis=1)
    s = np.arange(H) + 0.5  # distance to the lower effective wall
    exact = force / (2.0 * nu) * s * (H - s)
    l2 = np.linalg.norm(profile - exact) / np.linalg.norm(exact)

    s_fine = np.linspace(0.0, H, 400)
    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(10, 4.2))

    ax.plot(force / (2.0 * nu) * s_fine * (H - s_fine), s_fine, "-", color="0.3",
            label="analytical")
    ax.plot(profile, s, "o", ms=4, color="tab:blue", label=f"LBM {meta['version']}")
    ax.set_xlabel("$u_x$ (lattice units)")
    ax.set_ylabel("distance to wall $s$")
    ax.set_title(f"Poiseuille, H={H}, tau={tau:g}, real={meta['real']}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax_err.plot((profile - exact) / exact.max(), s, ".-", color="tab:red")
    ax_err.set_xlabel("$(u - u_{exact}) / u_{max}$")
    ax_err.set_title(f"relative L2 error = {l2:.2e}")
    ax_err.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useOffset=False)
    ax_err.grid(alpha=0.3)

    fig.tight_layout()
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=120)
    print(f"L2 relative error = {l2:.3e}  ->  {args.png}")


if __name__ == "__main__":
    main()
