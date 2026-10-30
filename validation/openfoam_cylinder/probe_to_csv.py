#!/usr/bin/env python3
"""Convert the OpenFOAM velocity probe into the probe.csv / meta.json format of the LBM runs,
so that scripts/strouhal.py computes the Strouhal number with exactly the same method.

Time is in units of D / U (D = 1, U = 1); it is written in the 'step' column.

Usage: probe_to_csv.py CASE_DIR OUTPUT_DIR
"""
import json
import re
import sys
from pathlib import Path


def main():
    case, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for line in (case / "postProcessing" / "probes" / "0" / "U").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"\s*(\S+)\s+\(([^)]*)\)", line)
        t = float(m.group(1))
        ux, uy, _ = (float(v) for v in m.group(2).split())
        rows.append((t, ux, uy))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "probe.csv", "w") as f:
        f.write("step,ux,uy,rho\n")
        for t, ux, uy in rows:
            f.write(f"{t:.6g},{ux:.8g},{uy:.8g},1\n")
    meta = {
        "version": "openfoam", "case": "cylinder", "real": "double", "re": 100,
        "diameter": 1, "u_in": 1, "steps": rows[-1][0], "solver": "foamRun incompressibleFluid",
    }
    (out / "meta.json").write_text(json.dumps(meta) + "\n")
    print(f"{len(rows)} probe samples, t = {rows[0][0]} .. {rows[-1][0]} -> {out}")


if __name__ == "__main__":
    main()
