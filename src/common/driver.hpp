#pragma once

// Command-line driver shared by every solver version.
//
// A Solver provides:
//   Solver(nx, ny, tau, force_x, force_y, Mask)
//   void init_equilibrium(rho, ux, uy);  void run(int steps);
//   void macroscopic(rho, ux, uy) const;  double total_mass() const;
//   const Mask& solid() const;  int threads() const;
// Optional (detected at compile time):
//   double timed_run(int steps)  -> elapsed seconds measured by the solver (CUDA events);
//                                   otherwise the host clock is used around run().
//   std::string bench_info() const -> extra JSON fields for the benchmark line (GPU, ...).
//
// Normal mode writes the fields to --out. Benchmark mode (--bench) writes nothing and
// prints one JSON line per timed repetition, prefixed by "BENCH ", for bench/run_bench.py.

#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "build_info.hpp"
#include "geometry.hpp"
#include "io.hpp"
#include "params.hpp"

// Target-specific compiler flags, reported with the global ones in benchmark mode.
#ifndef LBM_EXTRA_FLAGS
#define LBM_EXTRA_FLAGS ""
#endif

namespace lbm {

namespace detail {

inline double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

template <class S, class = void>
struct has_timed_run : std::false_type {};
template <class S>
struct has_timed_run<S, std::void_t<decltype(std::declval<S&>().timed_run(1))>>
    : std::true_type {};

template <class S, class = void>
struct has_bench_info : std::false_type {};
template <class S>
struct has_bench_info<S, std::void_t<decltype(std::declval<const S&>().bench_info())>>
    : std::true_type {};

// Elapsed seconds for `steps` steps: GPU solvers time themselves with CUDA events, because a
// kernel launch returns before the kernel has run.
template <class Solver>
double timed(Solver& solver, int steps) {
  if constexpr (has_timed_run<Solver>::value) {
    return solver.timed_run(steps);
  } else {
    const auto t0 = std::chrono::steady_clock::now();
    solver.run(steps);
    return seconds_since(t0);
  }
}

inline std::string compiler() {
#ifdef __CUDACC__
  return std::string(LBM_COMPILER) + " + nvcc " + std::to_string(__CUDACC_VER_MAJOR__) + "." +
         std::to_string(__CUDACC_VER_MINOR__) + "." + std::to_string(__CUDACC_VER_BUILD__);
#else
  return LBM_COMPILER;
#endif
}

inline std::string flags() {
  const std::string extra = LBM_EXTRA_FLAGS;
  return extra.empty() ? std::string(LBM_CXX_FLAGS) : std::string(LBM_CXX_FLAGS) + " " + extra;
}

template <class Solver>
void run_bench(Solver& solver, const Params& p, const char* version) {
  solver.run(p.warmup);
  for (int r = 0; r < p.reps; ++r) {
    const double t = timed(solver, p.steps);
    const double mlups = static_cast<double>(p.nx) * p.ny * p.steps / t / 1e6;
    std::string extra;
    if constexpr (has_bench_info<Solver>::value) extra = ", " + solver.bench_info();
    std::printf(
        "BENCH {\"version\": \"%s\", \"rep\": %d, \"nx\": %d, \"ny\": %d, \"steps\": %d, "
        "\"warmup\": %d, \"threads\": %d, \"time_s\": %.6f, \"mlups\": %.4f, \"real\": \"%s\", "
        "\"compiler\": \"%s\", \"flags\": \"%s\", \"build_type\": \"%s\"%s}\n",
        version, r, p.nx, p.ny, p.steps, p.warmup, solver.threads(), t, mlups, real_name,
        compiler().c_str(), flags().c_str(), LBM_BUILD_TYPE, extra.c_str());
    std::fflush(stdout);
  }
}

template <class Solver>
void run_and_write(Solver& solver, const Params& p, const char* version, real force) {
  const double mass0 = solver.total_mass();
  const double seconds = timed(solver, p.steps);

  // Indicative only: use --bench (warm-up and repetitions) for measurements.
  const double mlups = static_cast<double>(p.nx) * p.ny * p.steps / seconds / 1e6;
  std::printf("time=%.3f s  MLUPS=%.2f (indicative)  mass drift=%.3e\n", seconds, mlups,
              (solver.total_mass() - mass0) / mass0);

  std::vector<real> rho, ux, uy;
  solver.macroscopic(rho, ux, uy);
  const std::vector<real> vort = vorticity(ux, uy, solver.solid(), p.nx, p.ny);

  std::filesystem::create_directories(p.out_dir);
  const std::string dir = p.out_dir + "/";
  write_npy(dir + "rho.npy", rho, p.ny, p.nx);
  write_npy(dir + "ux.npy", ux, p.ny, p.nx);
  write_npy(dir + "uy.npy", uy, p.ny, p.nx);
  write_npy(dir + "vorticity.npy", vort, p.ny, p.nx);
  write_npy(dir + "solid.npy", solver.solid(), p.ny, p.nx);

  std::ofstream meta(dir + "meta.json");
  meta << "{\"version\": \"" << version << "\", \"case\": \"" << p.case_name
       << "\", \"nx\": " << p.nx << ", \"ny\": " << p.ny
       << ", \"tau\": " << static_cast<double>(p.tau)
       << ", \"umax\": " << static_cast<double>(p.u_max)
       << ", \"force_x\": " << static_cast<double>(force) << ", \"steps\": " << p.steps
       << ", \"real\": \"" << real_name << "\"}\n";
  std::printf("fields written to %s\n", p.out_dir.c_str());
}

}  // namespace detail

template <class Solver>
int run_driver(int argc, char** argv, const char* version) {
  try {
    const Params p = parse_args(argc, argv);
    if (p.case_name != "poiseuille") {
      std::fprintf(stderr, "unknown case '%s' (available: poiseuille)\n", p.case_name.c_str());
      return 1;
    }
    if (!p.bench) print_params(p);

    const int H = p.ny - 2;
    const real force = poiseuille_force(H, p.tau, p.u_max);
    Solver solver(p.nx, p.ny, p.tau, force, real(0), make_channel(p.nx, p.ny));
    solver.init_equilibrium(real(1), real(0), real(0));

    if (p.bench)
      detail::run_bench(solver, p, version);
    else
      detail::run_and_write(solver, p, version, force);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}

}  // namespace lbm
