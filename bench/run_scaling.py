#!/usr/bin/env python3
"""Strong and weak scaling of v4 (MPI) and the resulting efficiency.

strong: the global grid is fixed (--size), the number of ranks P grows;
weak:   the grid per rank is fixed (--size is per rank; ny grows with P).
Efficiency E(P) = MLUPS(P) / (P * MLUPS(1)) in both cases (ideal: 1).

A scaling measurement is only meaningful with one GPU per rank: the benchmark line reports
the number of distinct devices, and a row is marked valid_scaling = 0 when ranks share a GPU
(as on a single-GPU machine); scripts/plot_scaling.py refuses to plot such rows.

Examples:
  # single machine (correctness only, ranks share the GPU)
  python3 bench/run_scaling.py --exe build/src/v4_mpi_cuda/lbm_v4 --mode strong --np 1 2 4
  # cluster with Slurm, one GPU per task
  python3 bench/run_scaling.py --exe build/src/v4_mpi_cuda/lbm_v4 --launcher "srun --gpus-per-task=1" \\
      --np-flag -n --mode weak --size 4096x2048 --np 1 2 4 8
"""
import argparse
import csv
import datetime
import json
import os
import shlex
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COLUMNS = ["mode", "ranks", "devices", "valid_scaling", "backend", "gpu", "overlap", "nx", "ny",
           "steps", "reps", "mlups_mean", "mlups_std", "efficiency", "real", "flags", "launcher",
           "date", "commit"]


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
    ap.add_argument("--exe", required=True)
    ap.add_argument("--mode", choices=["strong", "weak"], required=True)
    ap.add_argument("--np", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--size", default="2048x2048", help="NXxNY (global if strong, per rank if weak)")
    ap.add_argument("--launcher", default="mpirun")
    ap.add_argument("--np-flag", default="-np")
    ap.add_argument("--mpi-flags", default="", help="extra launcher flags, e.g. --oversubscribe")
    ap.add_argument("--updates", type=float, default=5e8, help="cell updates per repetition (global)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    nx, ny = (int(v) for v in args.size.lower().split("x"))
    rows, base = [], None
    for p in sorted(args.np):
        gny = ny * p if args.mode == "weak" else ny
        steps = max(10, round(args.updates / (nx * gny)))
        cmd = (shlex.split(args.launcher) + [args.np_flag, str(p)] + shlex.split(args.mpi_flags)
               + [args.exe, "--bench", "--nx", str(nx), "--ny", str(gny), "--steps", str(steps),
                  "--warmup", str(max(1, steps // 5)), "--reps", str(args.reps)])
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, env=dict(os.environ)).stdout
        recs = [json.loads(l[len("BENCH "):]) for l in out.splitlines() if l.startswith("BENCH ")]
        if len(recs) != args.reps:
            raise RuntimeError(f"expected {args.reps} BENCH lines, got:\n{out}")
        mlups = [r["mlups"] for r in recs]
        mean = statistics.mean(mlups)
        if p == min(args.np):
            base = mean / p
        r0 = recs[0]
        valid = int(r0.get("backend") == "gpu" and r0.get("devices") == p)
        rows.append({
            "mode": args.mode, "ranks": p, "devices": r0.get("devices", 0), "valid_scaling": valid,
            "backend": r0.get("backend", ""), "gpu": r0.get("gpu", ""), "overlap": r0.get("overlap", ""),
            "nx": nx, "ny": gny, "steps": steps, "reps": args.reps,
            "mlups_mean": f"{mean:.2f}", "mlups_std": f"{statistics.stdev(mlups) if len(mlups) > 1 else 0:.2f}",
            "efficiency": f"{mean / (p * base):.3f}", "real": r0["real"], "flags": r0["flags"],
            "launcher": args.launcher, "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(),
        })
        note = "" if valid else "  (CPU engine or ranks sharing a GPU: not a GPU scaling measurement)"
        print(f"{args.mode} P={p}: {mean:.1f} MLUPS, efficiency {rows[-1]['efficiency']}{note}")

    gpu = rows[0]["gpu"].replace(" ", "-") or "cpu"
    path = args.csv or REPO / "results" / f"v4_{args.mode}_scaling_{gpu}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
