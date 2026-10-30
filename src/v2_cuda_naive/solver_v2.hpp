#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "d2q9.hpp"
#include "geometry.hpp"

namespace lbm {

// v2: naive CUDA port of v0, one thread per cell.
// Same numerics, same array-of-structures layout g[(y * nx + x) * 9 + i] and same push
// streaming as v0, in GPU global memory with two buffers swapped after each step.
// Compiled with --fmad=false: no fused multiply-add, so the results are bitwise identical to
// v0 (tested). The layout is kept on purpose: its cost is measured against v3 (SoA).
//
// Block size: 32 x 8 threads by default (a warp spans 32 consecutive cells along x);
// override with the environment variable LBM_BLOCK, e.g. LBM_BLOCK=16x16.
class SolverV2 {
 public:
  SolverV2(int nx, int ny, real tau, real force_x, real force_y, Mask solid);
  ~SolverV2();
  SolverV2(const SolverV2&) = delete;
  SolverV2& operator=(const SolverV2&) = delete;

  void init_equilibrium(real rho0, real ux0, real uy0);
  void step();
  void run(int steps);
  // Runs `steps` steps and returns the elapsed GPU time in seconds (CUDA events).
  double timed_run(int steps);

  // These copy the populations back to the host.
  void macroscopic(std::vector<real>& rho, std::vector<real>& ux, std::vector<real>& uy) const;
  double total_mass() const;
  std::vector<real> populations() const;

  int threads() const { return 0; }  // no CPU threads: see block and gpu in bench_info()
  // Extra JSON fields for the benchmark line: GPU, driver, CUDA versions, block size.
  std::string bench_info() const;

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  const Mask& solid() const { return solid_; }

 private:
  std::size_t cells() const { return static_cast<std::size_t>(nx_) * ny_; }

  int nx_, ny_;
  real tau_, fx_, fy_;
  Mask solid_;
  int block_x_ = 32, block_y_ = 8;
  // Device buffers (GPU global memory).
  real* g_d_ = nullptr;
  real* g_new_d_ = nullptr;
  std::uint8_t* solid_d_ = nullptr;
};

}  // namespace lbm
