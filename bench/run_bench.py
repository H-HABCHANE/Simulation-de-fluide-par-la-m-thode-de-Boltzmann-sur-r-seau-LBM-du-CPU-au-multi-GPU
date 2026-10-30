#!/usr/bin/env python3
"""Benchmark one solver version over several grid sizes (and thread counts).

Runs the executable in --bench mode (untimed warm-up, then timed repetitions inside the
process), and writes results/<version>_<machine>.csv with one row per (size, threads).

Examples:
  python3 bench/run_bench.py --exe build/src/v1_omp/lbm_v1 --threads 1 2 4 8
  python3 bench/run_bench.py --exe build/src/v2_cuda_naive/lbm_v2 --blocks 32x8 16x16
"""
import argparse
import csv
import datetime
import json
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

COLUMNS = [
    "version", "nx", "ny", "steps", "warmup", "reps", "threads",
    "time_mean_s", "time_std_s", "mlups_mean", "mlups_max", "mlups_std",
    "real", "cpu", "governor", "compiler", "flags", "build_type",
    "omp_proc_bind", "omp_places", "date", "commit",
    "gpu", "gpu_driver", "gpu_driver_cuda", "cuda_runtime", "cuda_compiled", "ecc", "block",
]
GPU_FIELDS = ["gpu", "gpu_driver_cuda", "cuda_runtime", "cuda_compiled", "ecc", "block"]


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def governor():
    try:
        return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        return "unknown"


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=REPO, capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "none"


def nvidia_driver():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.split()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return ""


def machine_tag(cpu):
    """Short, file-name friendly CPU name, e.g. 'i5-10310U'."""
    m = re.search(r"(i[3579]-\w+|Xeon\S*\s+\S+|EPYC\s+\S+|Ryzen\s+\d\s+\S+)", cpu)
    tag = m.group(1) if m else platform.node()
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", tag)


def parse_size(s):
    nx, ny = s.lower().split("x")
    return int(nx), int(ny)


def run_one(exe, nx, ny, steps, warmup, reps, threads, block, env):
    if threads is not None:
        env = dict(env, OMP_NUM_THREADS=str(threads))
    if block is not None:
        env = dict(env, LBM_BLOCK=block)
    cmd = [str(exe), "--bench", "--nx", str(nx), "--ny", str(ny), "--steps", str(steps),
           "--warmup", str(warmup), "--reps", str(reps)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True).stdout
    records = [json.loads(line[len("BENCH "):]) for line in out.splitlines()
               if line.startswith("BENCH ")]
    if len(records) != reps:
        raise RuntimeError(f"expected {reps} BENCH lines, got {len(records)}:\n{out}")
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--sizes", nargs="+", default=["128x128", "512x512", "1024x1024", "2048x2048"])
    ap.add_argument("--threads", nargs="*", type=int, default=None,
                    help="OMP_NUM_THREADS values to sweep (omit for sequential versions)")
    ap.add_argument("--blocks", nargs="*", default=None,
                    help="CUDA block sizes to sweep, e.g. 32x8 16x16 (sets LBM_BLOCK)")
    ap.add_argument("--updates", type=float, default=5e7,
                    help="target cell updates per repetition; steps = updates / (nx*ny)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--machine", default=None, help="tag used in the CSV file name")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument("--label", default=None,
                    help="value of the 'version' column (default: reported by the executable)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="explicit output file; rows are appended if it exists")
    args = ap.parse_args()

    cpu = cpu_model()
    env = dict(os.environ)
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    common = {
        "cpu": cpu, "governor": governor(), "commit": git_commit(),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "omp_proc_bind": env["OMP_PROC_BIND"], "omp_places": env["OMP_PLACES"],
    }

    rows = []
    for size in args.sizes:
        nx, ny = parse_size(size)
        steps = max(10, round(args.updates / (nx * ny)))
        warmup = max(1, steps // 5)
        for threads, block in [(t, b) for t in (args.threads or [None])
                               for b in (args.blocks or [None])]:
            recs = run_one(args.exe, nx, ny, steps, warmup, args.reps, threads, block, env)
            times = [r["time_s"] for r in recs]
            mlups = [r["mlups"] for r in recs]
            r0 = recs[0]
            row = {
                "version": args.label or r0["version"], "nx": nx, "ny": ny, "steps": steps, "warmup": warmup,
                "reps": args.reps, "threads": r0["threads"],
                "time_mean_s": f"{statistics.mean(times):.6f}",
                "time_std_s": f"{statistics.stdev(times) if len(times) > 1 else 0.0:.6f}",
                "mlups_mean": f"{statistics.mean(mlups):.3f}",
                "mlups_max": f"{max(mlups):.3f}",
                "mlups_std": f"{statistics.stdev(mlups) if len(mlups) > 1 else 0.0:.3f}",
                "real": r0["real"], "compiler": r0["compiler"], "flags": r0["flags"],
                "build_type": r0["build_type"], **common,
                **{k: r0.get(k, "") for k in GPU_FIELDS},
                "gpu_driver": nvidia_driver() if r0.get("gpu") else "",
            }
            rows.append(row)
            where = f"block={row['block']}" if row["gpu"] else f"threads={row['threads']}"
            print(f"{row['version']} {nx}x{ny} {where}: "
                  f"{row['mlups_mean']} +/- {row['mlups_std']} MLUPS")

    if args.csv:
        path = args.csv
        append = path.exists()
    else:
        tag = args.machine or (re.sub(r"[^A-Za-z0-9_.-]+", "-", rows[0]["gpu"]) if rows[0]["gpu"]
                               else machine_tag(cpu))
        path = args.out_dir / f"{rows[0]['version']}_{tag}.csv"
        append = False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not append:
            writer.writeheader()
        writer.writerows(rows)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
