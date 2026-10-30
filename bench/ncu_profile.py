#!/usr/bin/env python3
"""Profile one LBM kernel launch with Nsight Compute and summarise its memory behaviour.

Measured per kernel launch (one time step on an nx x ny grid):
- DRAM bytes read + written, divided by the number of cells, to compare with the ideal
  2 * 9 * sizeof(real) bytes per cell update (72 in float);
- global load/store sectors per request: 4 sectors (128 bytes) per warp request is a fully
  coalesced 32 x 4-byte access; more sectors means scattered accesses;
- DRAM throughput in % of peak, achieved occupancy.

Nsight Compute locks the clocks and flushes the caches before the profiled launch, so its
kernel duration is not used for MLUPS: timings come from run_bench.py. Profiling needs access
to the GPU performance counters; if they are blocked (ERR_NVGPUCTRPERM), nothing is written.

Example:
  python3 bench/ncu_profile.py --exe build-float/src/v3_cuda_opt/lbm_v3 --variant soa_pull_opt \\
      --label v3_soa_pull_opt --csv results/ncu_Tesla-T4.csv
"""
import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

METRICS = {
    "dram__bytes_read.sum": "dram_read_B",
    "dram__bytes_write.sum": "dram_write_B",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct_peak",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "ld_sectors",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "ld_requests",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": "st_sectors",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": "st_requests",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
}
COLUMNS = ["label", "nx", "ny", "real", "bytes_per_cell", "ideal_bytes_per_cell",
           "ld_sectors_per_request", "st_sectors_per_request", "dram_pct_peak",
           "achieved_occupancy_pct", "dram_read_B", "dram_write_B", "kernel"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--variant", default=None, help="LBM_V3_VARIANT for v3")
    ap.add_argument("--nx", type=int, default=2048)
    ap.add_argument("--ny", type=int, default=2048)
    ap.add_argument("--real-bytes", type=int, default=4, help="sizeof(real): 4 or 8")
    ap.add_argument("--csv", type=Path, required=True, help="output file (rows appended)")
    args = ap.parse_args()

    env = dict(os.environ)
    if args.variant:
        env["LBM_V3_VARIANT"] = args.variant
    cmd = ["ncu", "--csv", "--print-units", "base", "--kernel-name", "regex:step_kernel",
           "--launch-skip", "10", "--launch-count", "1",
           "--metrics", ",".join(METRICS),
           args.exe, "--bench", "--nx", str(args.nx), "--ny", str(args.ny),
           "--steps", "20", "--warmup", "0", "--reps", "1"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = proc.stdout + proc.stderr
    if "ERR_NVGPUCTRPERM" in out:
        sys.exit("ncu: access to GPU performance counters is blocked here (ERR_NVGPUCTRPERM); "
                 "use the effective bandwidth (MLUPS x bytes per cell) instead.")
    lines = [l for l in proc.stdout.splitlines() if l.startswith('"')]
    if not lines:
        sys.exit(f"ncu produced no metrics:\n{out[-2000:]}")

    values, kernel = {}, ""
    for row in csv.DictReader(io.StringIO("\n".join(lines))):
        name = row.get("Metric Name")
        if name in METRICS:
            values[METRICS[name]] = float(row["Metric Value"].replace(",", ""))
            kernel = row.get("Kernel Name", "")
    missing = [m for m in METRICS.values() if m not in values]
    if missing:
        sys.exit(f"missing metrics: {missing}")

    cells = args.nx * args.ny
    result = {
        "label": args.label, "nx": args.nx, "ny": args.ny,
        "real": "double" if args.real_bytes == 8 else "float",
        "bytes_per_cell": f"{(values['dram_read_B'] + values['dram_write_B']) / cells:.1f}",
        "ideal_bytes_per_cell": 2 * 9 * args.real_bytes,
        "ld_sectors_per_request": f"{values['ld_sectors'] / values['ld_requests']:.2f}",
        "st_sectors_per_request": f"{values['st_sectors'] / values['st_requests']:.2f}",
        "dram_pct_peak": f"{values['dram_pct_peak']:.1f}",
        "achieved_occupancy_pct": f"{values['achieved_occupancy_pct']:.1f}",
        "dram_read_B": int(values["dram_read_B"]), "dram_write_B": int(values["dram_write_B"]),
        "kernel": kernel[:60],
    }
    append = args.csv.exists()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if not append:
            w.writeheader()
        w.writerow(result)
    print(f"{args.label}: {result['bytes_per_cell']} B/cell (ideal {result['ideal_bytes_per_cell']}), "
          f"ld {result['ld_sectors_per_request']} / st {result['st_sectors_per_request']} sectors/req, "
          f"DRAM {result['dram_pct_peak']} % of peak, occupancy {result['achieved_occupancy_pct']} %")


if __name__ == "__main__":
    main()
