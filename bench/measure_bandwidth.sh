#!/usr/bin/env bash
# Measures the sustainable memory bandwidth with STREAM (J. D. McCalpin) and writes
# results/bandwidth_<machine>.csv. stream.c is downloaded into build/stream/ (not versioned).
#
# Usage: bench/measure_bandwidth.sh [machine_tag]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$REPO/build/stream"
URL="https://www.cs.virginia.edu/stream/FTP/Code/stream.c"
# Three arrays of 40M doubles = 960 MB in total, far larger than any CPU cache.
ARRAY_SIZE=40000000
NTIMES=20

mkdir -p "$WORK" "$REPO/results"
[ -f "$WORK/stream.c" ] || curl -fsSL "$URL" -o "$WORK/stream.c"
gcc -O3 -fopenmp -mcmodel=medium -DSTREAM_ARRAY_SIZE=$ARRAY_SIZE -DNTIMES=$NTIMES \
    "$WORK/stream.c" -o "$WORK/stream"

CPU="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')"
TAG="${1:-$(echo "$CPU" | grep -oE 'i[3579]-[A-Za-z0-9]+' || hostname)}"
GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
PHYS_CORES="$(lscpu -p=core | grep -v '^#' | sort -u | wc -l)"
LOGICAL="$(nproc)"
OUT="$REPO/results/bandwidth_${TAG}.csv"

echo "threads,copy_MBps,scale_MBps,add_MBps,triad_MBps,array_size,ntimes,cpu,governor,date" > "$OUT"
for T in $(printf '%s\n' 1 "$PHYS_CORES" "$LOGICAL" | sort -nu); do
  RES="$(OMP_NUM_THREADS=$T OMP_PROC_BIND=close OMP_PLACES=cores "$WORK/stream")"
  # "Best Rate MB/s" is the second column of each kernel line.
  rate() { echo "$RES" | awk -v k="$1:" '$1 == k { print $2 }'; }
  echo "$T,$(rate Copy),$(rate Scale),$(rate Add),$(rate Triad),$ARRAY_SIZE,$NTIMES,\"$CPU\",$GOV,$(date -Iseconds)" >> "$OUT"
  echo "threads=$T  Copy=$(rate Copy) MB/s  Triad=$(rate Triad) MB/s"
done
echo "-> $OUT"
