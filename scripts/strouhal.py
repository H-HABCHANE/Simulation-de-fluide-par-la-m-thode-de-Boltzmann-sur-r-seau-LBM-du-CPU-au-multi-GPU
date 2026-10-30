#!/usr/bin/env python3
"""Strouhal number of the cylinder wake from the velocity probe of a cylinder run.

The probe sits 3 D behind the cylinder centre, on its axis, where the transverse velocity
u_y oscillates at the vortex shedding frequency f. After discarding the start-up transient,
f is measured in two independent ways:
- FFT peak of u_y (Hann window, zero padding, parabolic refinement of the peak);
- mean period between upward zero crossings of u_y - mean(u_y).
St = f D / U, with D the cylinder diameter and U the inlet velocity (lattice units).

Reference: St = 0.164 at Re = 100 for an unconfined cylinder (Williamson, 1996). This domain
is periodic in y with a blockage D/H = 12.5 %, which speeds up the flow around the cylinder and
raises St slightly, so the check accepts 0.15 <= St <= 0.19 (a plausibility check, not a fine
validation). The row is appended to results/strouhal.csv.

The 'step' column may also be a time (OpenFOAM reference, validation/openfoam_cylinder): the
analysis only needs a constant sampling interval and D / U in the same units.

Usage: strouhal.py OUTPUT_DIR [--from 0.5] [--label v1]
"""
import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ST_MIN, ST_MAX = 0.15, 0.19


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--",
                                ".", ":(exclude)results"],
                               cwd=REPO, capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "none"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--from", dest="start", type=float, default=0.5,
                    help="fraction of the run discarded as transient (default 0.5)")
    ap.add_argument("--label", default=None, help="value of the 'run' column")
    ap.add_argument("--csv", type=Path, default=REPO / "results" / "strouhal.csv")
    args = ap.parse_args()

    meta = json.loads((args.output_dir / "meta.json").read_text())
    data = np.genfromtxt(args.output_dir / "probe.csv", delimiter=",", names=True)
    step, uy = data["step"], data["uy"]
    keep = step >= args.start * meta["steps"]
    step, uy = step[keep], uy[keep] - uy[keep].mean()
    dt = float(step[1] - step[0])  # steps between two samples
    D, U = meta["diameter"], meta["u_in"]

    # FFT peak with zero padding and parabolic interpolation around the maximum.
    n = len(uy)
    spec = np.abs(np.fft.rfft(uy * np.hanning(n), 16 * n))
    freqs = np.fft.rfftfreq(16 * n, d=dt)
    k = int(np.argmax(spec[1:]) + 1)
    a, b, c = np.log(spec[k - 1:k + 2] + 1e-300)
    shift = 0.5 * (a - c) / (a - 2 * b + c)
    f_fft = (k + shift) * (freqs[1] - freqs[0])

    # Mean period between upward zero crossings (linear interpolation of the crossing time).
    up = np.where((uy[:-1] < 0) & (uy[1:] >= 0))[0]
    t_cross = step[up] - uy[up] * (step[up + 1] - step[up]) / (uy[up + 1] - uy[up])
    periods = np.diff(t_cross)
    f_zero = 1.0 / periods.mean() if len(periods) else float("nan")

    st_fft, st_zero = f_fft * D / U, f_zero * D / U
    ok = ST_MIN <= st_fft <= ST_MAX and abs(st_fft - st_zero) <= 0.01
    print(f"{meta['version']}: St (FFT) = {st_fft:.4f}, St (zero crossings) = {st_zero:.4f} "
          f"over {len(periods)} periods; shedding period = {1 / f_fft:.4g} (in the unit of the step column); "
          f"u_y amplitude = {np.abs(uy).max():.4f}")
    print(f"[{'PASS' if ok else 'FAIL'}] {ST_MIN} <= St <= {ST_MAX} and both methods agree "
          f"within 0.01")

    row = {
        "run": args.label or meta["version"], "version": meta["version"], "real": meta["real"],
        # nx, ny and tau only exist for the LBM runs (empty for the OpenFOAM reference).
        "nx": meta.get("nx", ""), "ny": meta.get("ny", ""), "diameter": D, "u_in": U,
        "re": meta["re"], "tau": f"{meta['tau']:.5f}" if "tau" in meta else "",
        "steps": meta["steps"],
        "window_start": int(args.start * meta["steps"]),
        "st_fft": f"{st_fft:.4f}", "st_zero_crossing": f"{st_zero:.4f}",
        "periods": len(periods), "uy_amplitude": f"{np.abs(uy).max():.5f}",
        "commit": git_commit(), "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    append = args.csv.exists()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not append:
            w.writeheader()
        w.writerow(row)
    print(f"-> {args.csv}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
