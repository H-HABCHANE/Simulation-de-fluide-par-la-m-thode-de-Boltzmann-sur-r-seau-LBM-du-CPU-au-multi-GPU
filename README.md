# LBM D2Q9 fluid solver — from CPU to multi-GPU

A 2D Lattice Boltzmann (D2Q9, BGK collision) fluid solver, written in C++17 and CUDA,
developed in successive versions that are all kept, validated against each other and
benchmarked on the same metric (MLUPS, million lattice updates per second).

> **Status: work in progress.** No performance numbers are published yet. Every number
> that will appear here comes from a `results/*.csv` file produced by the benchmark script,
> with the exact hardware, driver, compiler flags, grid size and step count.

## Versions

| Version | Description | Status |
|---|---|---|
| v0 | Sequential CPU — correctness reference | planned |
| v1 | CPU OpenMP — honest performance baseline | planned |
| v2 | Naive CUDA — one thread per cell | planned |
| v3 | Optimized CUDA — SoA layout, fused collide + stream, double buffering | planned |
| v4 | Multi-GPU MPI + CUDA — domain decomposition, halo exchange, overlap | planned |

## Validation (planned)

- Poiseuille flow: velocity profile compared with the analytical parabola (relative L2 error).
- Cross-version comparison: every version must reproduce v0 within a justified tolerance.

## Build

```bash
cmake -S . -B build                      # CPU only, real = float
cmake --build build -j
ctest --test-dir build --output-on-failure
```

| CMake option | Default | Effect |
|---|---|---|
| `LBM_ENABLE_CUDA` | `OFF` | Build the CUDA versions (default arch: `sm_75`, Tesla T4) |
| `LBM_ENABLE_MPI` | `OFF` | Build the MPI + CUDA version (requires CUDA) |
| `LBM_REAL_DOUBLE` | `OFF` | Use `double` instead of `float` |

### Docker

```bash
docker build -t lbm-d2q9 -f docker/Dockerfile .
docker run --rm -it --gpus all -v "$PWD":/workspace lbm-d2q9
```

## Toolchain

- C++17, CMake ≥ 3.20, OpenMP, Open MPI.
- CUDA toolkit: **12.4** (Docker image `nvidia/cuda:12.4.1-devel-ubuntu22.04`), chosen to be
  no newer than the toolkit and driver available on Google Colab. To be confirmed against
  `nvcc --version` and `nvidia-smi` on Colab before the first GPU measurements.
