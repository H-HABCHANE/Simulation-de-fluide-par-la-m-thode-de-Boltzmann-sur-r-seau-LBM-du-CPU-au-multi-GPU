#!/usr/bin/env python3
"""Memory-bandwidth ceiling in MLUPS for the D2Q9 LBM, from a STREAM CSV.

Each cell update reads 9 populations and writes 9: 2 * 9 * sizeof(real) bytes.

- STREAM ceiling: best Copy rate / (2 * 9 * sizeof(real)). STREAM Copy reports
  (bytes read + bytes written) / time and, like the LBM, pays the hidden write-allocate
  read on the CPU, so the two are directly comparable.
- Hardware ceiling (optional, --peak-gbs from the memory spec): the DRAM peak must also
  carry the write-allocate traffic, so 3 * 9 * sizeof(real) bytes per cell are used.

Usage: ceiling.py results/bandwidth_<machine>.csv [--peak-gbs 42.7]
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
    args = ap.parse_args()

    with open(args.bandwidth_csv) as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r["copy_MBps"]))
    copy_bps = float(best["copy_MBps"]) * 1e6

    print(f"STREAM best Copy: {copy_bps / 1e9:.2f} GB/s ({best['threads']} threads)")
    for name, size in SIZES.items():
        line = f"  {name:6s}: STREAM ceiling = {copy_bps / (2 * Q * size) / 1e6:7.1f} MLUPS"
        if args.peak_gbs:
            line += f"   hardware ceiling = {args.peak_gbs * 1e9 / (3 * Q * size) / 1e6:7.1f} MLUPS"
        print(line)


if __name__ == "__main__":
    main()
