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
//   void set_inflow_outflow(real u_in), void populations_at(x, y, out) -> cylinder case.
//   bool is_root() const          -> MPI solvers: only the root rank prints and writes files
//                                    (all ranks run the driver and its collective calls).
//
// Cases: --case poiseuille (periodic channel driven by a body force) and --case cylinder
// (flow past a cylinder: inlet velocity --umax, Reynolds number --re, diameter --diameter;
// tau is derived from them). The cylinder case writes a velocity probe (probe.csv) and,
// with --frame-every, vorticity frames (frames/vort_<step>.npy).
//
// Normal mode writes the fields to --out. Benchmark mode (--bench) writes nothing and
// prints one JSON line per timed repetition, prefixed by "BENCH ", for bench/run_bench.py.

#include <algorithm>
#include <chrono>
#include <cmath>
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
struct has_open_boundaries : std::false_type {};
template <class S>
struct has_open_boundaries<
    S, std::void_t<decltype(std::declval<S&>().set_inflow_outflow(real(0))),
                   decltype(std::declval<const S&>().populations_at(0, 0, nullptr))>>
    : std::true_type {};

template <class S, class = void>
struct has_is_root : std::false_type {};
template <class S>
struct has_is_root<S, std::void_t<decltype(std::declval<const S&>().is_root())>>
    : std::true_type {};

template <class Solver>
bool is_root(const Solver& s) {
  if constexpr (has_is_root<Solver>::value) return s.is_root();
  else return true;
}

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
    if (!is_root(solver)) continue;
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
  const double mass1 = solver.total_mass();
  std::vector<real> rho, ux, uy;
  solver.macroscopic(rho, ux, uy);
  if (!is_root(solver)) return;
  std::printf("time=%.3f s  MLUPS=%.2f (indicative)  mass drift=%.3e\n", seconds, mlups,
              (mass1 - mass0) / mass0);
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

// Geometry of the cylinder case: centre at x = 5 D, one cell above mid-height (breaks the
// symmetry deterministically so that vortex shedding starts); probe 3 D behind the centre.
struct CylinderSetup {
  double xc, yc;
  int probe_x, probe_y;
  real tau;
};

inline CylinderSetup cylinder_setup(const Params& p) {
  CylinderSetup c{};
  c.xc = 5.0 * p.diameter;
  c.yc = p.ny / 2 + 1;
  c.probe_x = static_cast<int>(c.xc + 3.0 * p.diameter);
  c.probe_y = static_cast<int>(c.yc);
  const double nu = static_cast<double>(p.u_max) * p.diameter / p.re;
  c.tau = static_cast<real>(0.5 + 3.0 * nu);
  return c;
}

template <class Solver>
void run_cylinder(Solver& solver, const Params& p, const char* version, const CylinderSetup& cs) {
  using d2q9::Q;
  const bool root = is_root(solver);
  const std::string dir = p.out_dir + "/";
  std::ofstream probe;
  if (root) {
    std::filesystem::create_directories(p.out_dir + "/frames");
    probe.open(dir + "probe.csv");
    probe << "step,ux,uy,rho\n";
  }

  double seconds = 0.0;
  int n = 0, frames = 0;
  while (n < p.steps) {
    const int chunk = std::min(p.probe_every, p.steps - n);
    seconds += timed(solver, chunk);
    n += chunk;

    real f[Q];
    solver.populations_at(cs.probe_x, cs.probe_y, f);
    real m0 = 0, mx = 0, my = 0;
    for (int i = 0; i < Q; ++i) {
      m0 += f[i];
      mx += d2q9::cx[i] * f[i];
      my += d2q9::cy[i] * f[i];
    }
    const real rho = real(1) + m0;
    if (root)
      probe << n << ',' << static_cast<double>(mx / rho) << ',' << static_cast<double>(my / rho)
            << ',' << static_cast<double>(rho) << '\n';
    if (!std::isfinite(static_cast<double>(rho)))
      throw std::runtime_error("the simulation diverged (non-finite density at the probe)");

    if (p.frame_every > 0 && n >= p.frame_start && n % p.frame_every == 0) {
      std::vector<real> r, ux, uy;
      solver.macroscopic(r, ux, uy);
      char name[64];
      std::snprintf(name, sizeof name, "frames/vort_%07d.npy", n);
      if (root)
        write_npy(dir + name, vorticity(ux, uy, solver.solid(), p.nx, p.ny, false), p.ny, p.nx);
      ++frames;
    }
  }
  const double mlups = static_cast<double>(p.nx) * p.ny * p.steps / seconds / 1e6;
  std::vector<real> rho, ux, uy;
  solver.macroscopic(rho, ux, uy);
  if (!root) return;
  std::printf("time=%.3f s  MLUPS=%.2f (indicative, without I/O)  frames=%d\n", seconds, mlups,
              frames);
  write_npy(dir + "rho.npy", rho, p.ny, p.nx);
  write_npy(dir + "ux.npy", ux, p.ny, p.nx);
  write_npy(dir + "uy.npy", uy, p.ny, p.nx);
  write_npy(dir + "vorticity.npy", vorticity(ux, uy, solver.solid(), p.nx, p.ny, false), p.ny, p.nx);
  write_npy(dir + "solid.npy", solver.solid(), p.ny, p.nx);

  std::ofstream meta(dir + "meta.json");
  meta << "{\"version\": \"" << version << "\", \"case\": \"cylinder\", \"nx\": " << p.nx
       << ", \"ny\": " << p.ny << ", \"tau\": " << static_cast<double>(cs.tau)
       << ", \"u_in\": " << static_cast<double>(p.u_max) << ", \"re\": " << p.re
       << ", \"diameter\": " << p.diameter << ", \"xc\": " << cs.xc << ", \"yc\": " << cs.yc
       << ", \"probe_x\": " << cs.probe_x << ", \"probe_y\": " << cs.probe_y
       << ", \"probe_every\": " << p.probe_every << ", \"frame_every\": " << p.frame_every
       << ", \"steps\": " << p.steps << ", \"real\": \"" << real_name << "\"}\n";
  std::printf("probe, frames and fields written to %s\n", p.out_dir.c_str());
}

}  // namespace detail

template <class Solver>
int run_driver(int argc, char** argv, const char* version) {
  try {
    const Params p = parse_args(argc, argv);
    if (p.case_name == "cylinder") {
      if constexpr (detail::has_open_boundaries<Solver>::value) {
        const detail::CylinderSetup cs = detail::cylinder_setup(p);
        Solver solver(p.nx, p.ny, cs.tau, real(0), real(0),
                      make_cylinder(p.nx, p.ny, cs.xc, cs.yc, p.diameter));
        if (detail::is_root(solver))
          std::printf("case=cylinder nx=%d ny=%d D=%d Re=%g u_in=%g -> tau=%.4f, Ma=%.3f, "
                    "steps=%d real=%s out=%s\n",
                    p.nx, p.ny, p.diameter, p.re, static_cast<double>(p.u_max),
                    static_cast<double>(cs.tau), std::sqrt(3.0) * static_cast<double>(p.u_max),
                    p.steps, real_name, p.out_dir.c_str());
        solver.set_inflow_outflow(p.u_max);
        solver.init_equilibrium(real(1), p.u_max, real(0));
        detail::run_cylinder(solver, p, version, cs);
        return 0;
      } else {
        std::fprintf(stderr, "%s does not implement the open boundaries of the cylinder case\n",
                     version);
        return 1;
      }
    }
    if (p.case_name != "poiseuille") {
      std::fprintf(stderr, "unknown case '%s' (available: poiseuille, cylinder)\n",
                   p.case_name.c_str());
      return 1;
    }
    const int H = p.ny - 2;
    const real force = poiseuille_force(H, p.tau, p.u_max);
    Solver solver(p.nx, p.ny, p.tau, force, real(0), make_channel(p.nx, p.ny));
    if (!p.bench && detail::is_root(solver)) print_params(p);
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
