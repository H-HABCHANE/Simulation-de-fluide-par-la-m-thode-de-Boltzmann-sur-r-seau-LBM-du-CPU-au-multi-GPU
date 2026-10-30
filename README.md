# LBM D2Q9 fluid solver — from CPU to multi-GPU

[![CI](https://github.com/H-HABCHANE/Simulation-de-fluide-par-la-m-thode-de-Boltzmann-sur-r-seau-LBM-du-CPU-au-multi-GPU/actions/workflows/ci.yml/badge.svg)](https://github.com/H-HABCHANE/Simulation-de-fluide-par-la-m-thode-de-Boltzmann-sur-r-seau-LBM-du-CPU-au-multi-GPU/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/H-HABCHANE/Simulation-de-fluide-par-la-m-thode-de-Boltzmann-sur-r-seau-LBM-du-CPU-au-multi-GPU/blob/main/notebooks/run_colab.ipynb)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus)
![OpenMP](https://img.shields.io/badge/OpenMP-parallel-informational)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia)
![MPI](https://img.shields.io/badge/MPI-Open%20MPI-informational)
![CMake](https://img.shields.io/badge/build-CMake-064F8C?logo=cmake)

A 2D lattice Boltzmann solver written five times, from a sequential reference to multi-GPU:
every version is validated bit for bit against the first one and benchmarked against the
memory-bandwidth ceiling of its machine.

![Von Karman vortex street behind a cylinder, Re = 100](docs/karman.gif)

*Vorticity behind a cylinder at Re = 100 computed with this solver (blue: clockwise, red:
counter-clockwise), 4 vortex-shedding periods.*

## Key results

- **99 % of the GPU memory-bandwidth ceiling**: the optimised CUDA kernel (v3) runs at
  3,187 million lattice updates per second (MLUPS) on a Tesla T4, 2.2x the naive port (v2),
  with the gain explained by Nsight Compute measurements.
- **Bit-for-bit identical results** across all versions: sequential, OpenMP (1 to 8 threads),
  CUDA, and MPI + CUDA (1 to 4 ranks) produce exactly the same numbers as the reference.
- **Validated physics**: second-order convergence on Poiseuille flow (observed order 2.000);
  flow past a cylinder at Re = 100 with Strouhal number 0.1807, within 0.8 % of an independent
  OpenFOAM (finite-volume) simulation of the same domain.
- **Measured, not assumed**: every performance and flow number in this README comes from a
  `results/*.csv` file that records the hardware, driver, compiler, flags, grid size, step count
  and git commit; accuracy numbers come from the tests (`ctest`).

## Results at a glance

![MLUPS vs memory-bandwidth ceiling](docs/mlups_vs_ceiling.png)

Float, 2048 x 2048 grid. The ceiling of each machine is its measured memory bandwidth divided
by the 72 bytes moved per cell update (9 populations read, 9 written).

| Version | Machine | MLUPS | % of ceiling | Source |
|---|---|---|---|---|
| v0 sequential reference | Intel i5-10310U, 1 core | 12.4 | 6 % | `results/v0_i5-10310U.csv` |
| v1_naive, OpenMP port of v0 | i5-10310U, 4 threads | 30.5 | 15 % | `results/v1_naive_i5-10310U.csv` |
| v1, OpenMP, pull + SoA + SIMD | i5-10310U, 4 threads | 110.8 | 56 % (of 199) | `results/v1_i5-10310U.csv` |
| v2, naive CUDA (AoS, push) | Tesla T4 | 1,446 | 45 % | `results/v3_steps_Tesla-T4.csv` |
| v3, CUDA, SoA + pull, no modulo | Tesla T4 | 3,187 | 99 % (of 3,221) | `results/v3_steps_Tesla-T4.csv` |

- CPU ceiling: STREAM Copy, 14.3 GB/s. GPU ceiling: best of a copy kernel and
  `cudaMemcpy` device-to-device, 231.9 GB/s (ECC on); the T4 datasheet gives 320 GB/s.
- v2 and v3 were measured in the same session: GPU results vary by about 5 % between sessions
  on the shared cloud machine.
- CPU and GPU are two different machines: v3 is 29x faster than v1 on this pair, mostly because
  the T4 has 16x the memory bandwidth of the laptop.

## Versions

| Version | What it is | Code |
|---|---|---|
| v0 | Sequential, simple and readable: the correctness reference | `src/v0_cpu/` |
| v1 | OpenMP; v1_naive kept as the starting point of the optimisation steps | `src/v1_omp/` |
| v2 | Naive CUDA, one thread per cell, same layout as v0 | `src/v2_cuda_naive/` |
| v3 | Optimised CUDA; the optimisation steps are kept as run-time variants | `src/v3_cuda_opt/` |
| v4 | MPI + CUDA, one rank per GPU, strips of rows with halo exchange and overlap | `src/v4_mpi_cuda/` |

Model: D2Q9 lattice, BGK collision, Guo body force, halfway bounce-back walls, populations
stored as `f - w` (see "Validation"). C++17, CUDA 13.0, Open MPI, CMake.

## Validation

**Physics.** Plane Poiseuille flow, compared with the analytical parabola by `ctest` in float
and double (`tests/poiseuille_check.hpp`, run for v0, v2 and v3):

![Poiseuille profile](docs/poiseuille_profile.png)

| Check | double | float |
|---|---|---|
| A. `tau = 1/2 + sqrt(3)/4` (BGK + bounce-back is exact for this tau) | L2 = 1.3e-11 (limit 1e-8) | limit 2e-4 |
| B. `tau = 1`, H = 16, 32, 64 | observed order 2.000 (limits 1.9-2.1) | L2 < 2e-3 |
| C. Mass conservation (relative drift) | < 1e-13 | < 1e-6 |

Populations are stored shifted by their rest value (`f_i - w_i`): in single precision this
keeps the float resolution on the flow-dependent part of `f`; without it the float run
stagnates far from the steady state. The float stagnation error (~1e-4) is still as large as
the H = 64 discretisation error, so the convergence order is only checked in double.

**Cross-version equality.** Each version must reproduce v0 bit for bit after 500 steps of a
non-uniform case (channel with an obstacle, force in x and y, grid 53 x 31 that is not a
multiple of the CUDA block size) and on the cylinder case:

| Version | Configurations tested | Test |
|---|---|---|
| v1, v1_naive | 1, 2, 3, 4, 8 threads | `tests/test_v1_vs_v0.cpp`, `tests/test_cylinder_v1.cpp` |
| v2 | Tesla T4, `--fmad=false` | `tests/test_v2_vs_v0.cu` |
| v3 | 3 kernel variants, Tesla T4 | `tests/test_v3_vs_v0.cu`, `tests/test_cylinder_v3.cu` |
| v4 | 1, 2, 3, 4 MPI ranks, with and without overlap, CPU and GPU engines | `tests/test_v4_vs_v0.cpp` |

Bitwise equality holds because every version performs the same floating-point operations in
the same order (no fused multiply-add: `--fmad=false` on the GPU, `-ffp-contract=off` on the
CPU). The tests were checked by mutation: breaking the outlet rule or the halo exchange on
purpose makes them fail. `compute-sanitizer` reports no error on v2 and v3
(memcheck, initcheck, racecheck, synccheck) and on v4 with 2 ranks (memcheck, initcheck,
synccheck).

**Continuous integration** (6 jobs on every push): CPU build and tests in float and double; MPI
build and tests with the v4 CPU engine (1 to 4 ranks) in float and double; CUDA build with
CUDA 13.0 in float and double. GitHub runners have no GPU: the GPU tests run on Colab.

## Optimisation steps

Each step was measured on its own, and kept only if the measurement justified it.

**CPU** (`results/v1_steps_i5-10310U.csv`, 2048 x 2048, one thread; the 4-thread values of
the first steps were affected by background load and are not quoted):

| Step | MLUPS | What changed |
|---|---|---|
| 0. v1_naive | 11.6 | v0 + `#pragma omp parallel for` |
| 1. `__restrict__` local pointers | 11.9 | no reload of members through `this` |
| 2. no integer modulo | 16.6 | 18 divisions per cell removed from the neighbour lookup |
| 3. pull + SoA + SIMD | 47.4 | contiguous accesses; vectorisation unblocked (`-fopt-info-vec-all`) |
| 4. `-march=native -ffp-contract=off` | 72.1 | AVX2 without FMA (FMA: 74.2, not kept: breaks bitwise equality) |

Before step 3 the code was compute-bound (its speed hardly changed between a grid that fits
in cache and one that does not); after it, v1 is memory-bound on large grids.

**GPU** (`results/v3_steps_Tesla-T4.csv` and `results/ncu_Tesla-T4.csv`, 2048 x 2048, same
session; Nsight Compute sectors per request: 4 = fully coalesced warp access):

| Kernel | MLUPS | % of ceiling | DRAM bytes / cell | Load / store sectors per request |
|---|---|---|---|---|
| v2: AoS, push | 1,446 | 45 % | 141.8 | 21.1 / 32.0 |
| v3 soa_push: structure of arrays | 2,916 | 91 % | 101.3 | 2.7 / 4.7 |
| v3 soa_pull: pull streaming | 3,025 | 94 % | 96.2 | 3.1 / 4.0 |
| v3 soa_pull_opt: no integer modulo | 3,187 | 99 % | 94.1 | 3.1 / 4.0 |

In v2 every 4-byte store of a warp lands in a different sector (array of structures + push);
the structure-of-arrays layout makes the accesses coalesced, pull streaming makes every store a
fully aligned 128-byte write, and removing the modulo cuts the remaining instruction overhead.

## Flow past a cylinder (Re = 100)

1000 x 320 grid, cylinder diameter D = 40 cells, inlet velocity 0.05 (Mach 0.087),
tau = 0.56, 160 000 steps. Inlet: equilibrium at the inlet velocity; outlet: zero gradient;
periodic in y (blockage 12.5 %). The cylinder is placed one cell above mid-height, which
triggers the vortex shedding deterministically.

| Run | Strouhal number (FFT / zero crossings) | Probe amplitude u_y / U | Source |
|---|---|---|---|
| v1, CPU | 0.1807 / 0.1805 | 0.646 | `results/strouhal.csv` |
| v3, Tesla T4 | 0.1807 / 0.1805 | 0.646 | `results/strouhal.csv` |
| v4, 2 MPI ranks on the Tesla T4 | 0.1807 / 0.1805 | 0.646 | `results/strouhal.csv` |
| OpenFOAM, finite volumes, smooth cylinder (independent reference) | 0.1821 / 0.1821 | 0.649 | `results/strouhal.csv` |

All runs are analysed by the same script over the same time window (15 shedding periods).
The LBM agrees with OpenFOAM within 0.8 % on the Strouhal number and 0.4 % on the amplitude;
the remaining difference is consistent with the staircase cylinder of the LBM and the
different inlet and outlet conditions. For comparison, an unconfined cylinder gives
St = 0.164 at Re = 100 (Williamson, 1996): the blockage of this periodic domain raises St.
The run is deterministic: re-running it from the recorded commit reproduces the probe signal
bit for bit.

## Multi-GPU version (v4)

One MPI rank per GPU; the domain is split into strips of rows (rows are contiguous in memory,
so a halo row is one contiguous block per population, and the inlet and outlet columns stay
local to every rank). After each step only the 3 populations that cross a strip boundary are
exchanged: those going up (`i = 2, 5, 6`) and those going down (`i = 4, 7, 8`), i.e. 3 x nx
values per message, through page-locked host buffers. Overlap: the two boundary rows are
computed first, then the interior rows run in a second CUDA stream while the halos are copied
and exchanged (`LBM_V4_OVERLAP=0` disables it). The same cell update
(`src/v4_mpi_cuda/strip_update.hpp`) is compiled for a GPU engine and for a CPU engine, which
tests the MPI logic in CI without a GPU.

**No scaling result is published**: a scaling measurement needs one GPU per rank, and only a
single GPU was available. `bench/run_scaling.py` (strong and weak scaling) records the number of
distinct GPUs and marks runs where ranks share a device as invalid; `scripts/plot_scaling.py`
refuses to plot them.

## Reproduce

**GPU, in one click**: the [Colab notebook](notebooks/run_colab.ipynb) (badge above) builds
everything on a Tesla T4, runs all tests and `compute-sanitizer`, and can redo every GPU
measurement; flags at the top choose what to run.

**Local build (CPU)**:

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/src/v1_omp/lbm_v1 --case cylinder --nx 1000 --ny 320 --diameter 40 --re 100 --umax 0.05 \
    --steps 160000 --frame-every 450 --frame-start 142000 --out output/cylinder
python3 scripts/strouhal.py output/cylinder --from 0.55   # Strouhal number
python3 scripts/make_video.py output/cylinder             # docs/karman.gif
```

**Docker** (CUDA 13.0, Open MPI, Python; the source tree is mounted, not copied):

```bash
docker build -t lbm-d2q9 -f docker/Dockerfile .
docker run --rm -it --gpus all -v "$PWD":/workspace lbm-d2q9
```

**Benchmarks**:

```bash
bench/measure_bandwidth.sh                                   # STREAM -> results/bandwidth_<cpu>.csv
python3 bench/ceiling.py results/bandwidth_<cpu>.csv         # MLUPS ceiling
python3 bench/run_bench.py --exe build/src/v1_omp/lbm_v1 --threads 1 2 4 8
python3 bench/run_bench.py --exe build/src/v3_cuda_opt/lbm_v3 --variants soa_push soa_pull soa_pull_opt
python3 bench/ncu_profile.py --exe build/src/v3_cuda_opt/lbm_v3 --label v3 --csv results/ncu.csv
```

Each benchmark run does an untimed warm-up, then 5 timed repetitions inside the process (CUDA
events on the GPU) and reports mean and standard deviation. MLUPS = nx * ny * steps / time / 1e6.

**OpenFOAM reference**: `validation/openfoam_cylinder/Allrun` (OpenFOAM 14 and gmsh).

## Limitations and next steps

- Multi-GPU scaling is not measured yet (single GPU available); the script is ready.
- With one rank, v4 pays a fixed cost for the decomposition (stream synchronisation, halos
  through the host) compared with v3; a device-to-device exchange would remove it.
- Nsight Compute measures 94 DRAM bytes per cell for v3 against 72 ideal; the cause (ECC,
  L2 reuse of neighbouring rows) is not identified yet.
- Double precision on the T4 is limited by its FP64 throughput, not by memory.
- Physics: BGK collision (TRT or MRT would allow lower viscosities), staircase cylinder
  (interpolated bounce-back would be second order on curved walls), simple equilibrium inlet.
- CPU measurements come from a laptop (turbo, power limit): they are reproducible within a
  few percent on an idle machine on mains power, which the CSV files document.

## Repository layout

```
src/common/     parameters, lattice constants, boundaries, I/O, shared driver
src/v0_cpu/  src/v1_omp/  src/v2_cuda_naive/  src/v3_cuda_opt/  src/v4_mpi_cuda/
tests/          Poiseuille, cross-version and MPI tests (CTest)
bench/          benchmark, bandwidth, Nsight Compute and scaling scripts
scripts/        plots, Strouhal number, vortex-street animation
results/        all measurements (CSV), one file per machine and experiment
validation/     OpenFOAM reference case of the cylinder flow
notebooks/      Colab notebook (Tesla T4)
docker/         CUDA 13.0 build environment
docs/           figures used in this README
```

| CMake option | Default | Effect |
|---|---|---|
| `LBM_ENABLE_CUDA` | `OFF` | Build the CUDA versions (default arch: `sm_75`, Tesla T4) |
| `LBM_ENABLE_MPI` | `OFF` | Build v4 (MPI): GPU engine with CUDA, CPU engine without |
| `LBM_REAL_DOUBLE` | `OFF` | Use `double` instead of `float` |
| `LBM_NATIVE` | `OFF` | `-march=native -ffp-contract=off` (vectorisation for the build CPU, no FMA) |
| `LBM_CUDA_FMAD` | `OFF` | Allow FMA contraction in CUDA kernels (speed measurement only) |
| `LBM_MPIEXEC_FLAGS` | empty | Extra `mpiexec` flags for the MPI tests, e.g. `--oversubscribe` |

Run-time selection: `LBM_BLOCK=32x8` (CUDA block size), `LBM_V3_VARIANT=soa_push|soa_pull|soa_pull_opt`,
`LBM_V4_OVERLAP=0|1`.
