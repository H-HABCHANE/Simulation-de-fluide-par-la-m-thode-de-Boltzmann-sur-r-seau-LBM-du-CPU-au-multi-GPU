#!/usr/bin/env python3
"""Memory-bandwidth ceiling in MLUPS for the D2Q9 LBM, from a bandwidth CSV.

Accepts the CPU file of measure_bandwidth.sh (STREAM) or the GPU file of gpu_bandwidth
(copy kernel); both have a copy_MBps column.

Each cell update reads 9 populations and writes 9: 2 * 9 * sizeof(real) bytes.

- STREAM ceiling: best Copy rate / (2 * 9 * sizeof(real)). STREAM Copy reports
  (bytes read + bytes written) / time and, like the LBM, pays the hidden write-allocate
  read on the CPU, so the two are directly comparable.
- Hardware ceiling (optional, --peak-gbs from the memory spec): the DRAM peak must also
  carry the write-allocate traffic, so 3 * 9 * sizeof(real) bytes per cell are used.

On the GPU, a write that covers a whole 32-byte sector (coalesced writes) is not read from
memory first; --gpu makes the hardware ceiling use 2 * 9 * sizeof(real) bytes. The measured
ceiling (copy kernel) is the reference in both cases.

Usage: ceiling.py results/bandwidth_<machine>.csv [--peak-gbs 42.7] [--gpu]
"""
import argparse
import csv

Q = 9
SIZES = {"float": 4, "double": 8}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bandwidth_csv")
    ap.add_argument("--peak-gbs", type=float, default=None,
                    help="theoretical DRAM bandwidth in GB/s (channels * 8 B * MT/s)")
    ap.add_argument("--gpu", action="store_true",
                    help="GPU: no write-allocate read in the hardware ceiling (coalesced writes)")
    args = ap.parse_args()

    with open(args.bandwidth_csv) as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r["copy_MBps"]))
    copy_bps = float(best["copy_MBps"]) * 1e6

    where = f"{best['threads']} threads" if "threads" in best else best.get("gpu", "GPU")
    print(f"Measured best copy: {copy_bps / 1e9:.2f} GB/s ({where})")
    peak_factor = 2 if args.gpu else 3
    for name, size in SIZES.items():
        line = f"  {name:6s}: measured ceiling = {copy_bps / (2 * Q * size) / 1e6:7.1f} MLUPS"
        if args.peak_gbs:
            hw = args.peak_gbs * 1e9 / (peak_factor * Q * size) / 1e6
            line += f"   hardware ceiling = {hw:7.1f} MLUPS"
        print(line)


if __name__ == "__main__":
    main()
