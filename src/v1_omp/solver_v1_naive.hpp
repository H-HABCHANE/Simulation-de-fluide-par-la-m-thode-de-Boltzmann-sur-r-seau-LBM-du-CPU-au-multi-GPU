#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "d2q9.hpp"
#include "geometry.hpp"

namespace lbm {

// v1_naive: v0 parallelised with OpenMP over the rows y, nothing else changed.
// Kept as the starting point of the CPU optimisation steps (see solver_v1.hpp).
// Same numerics, same layout, same loop body as v0, so the results are bitwise identical.
// Push streaming is race-free: each (cell, direction) slot of g_new is written by exactly one
// source cell per step (by streaming or by bounce-back).
class SolverV1Naive {
 public:
  SolverV1Naive(int nx, int ny, real tau, real force_x, real force_y, Mask solid);

  void init_equilibrium(real rho0, real ux0, real uy0);
  void step();
  void run(int steps);

  void macroscopic(std::vector<real>& rho, std::vector<real>& ux, std::vector<real>& uy) const;
  double total_mass() const;

  std::vector<real> populations() const;
  int threads() const;

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  const Mask& solid() const { return solid_; }

 private:
  std::size_t cell(int x, int y) const { return static_cast<std::size_t>(y) * nx_ + x; }
  std::size_t size() const { return static_cast<std::size_t>(nx_) * ny_ * d2q9::Q; }
  void moments(std::size_t c, real& drho, real& ux, real& uy) const;

  int nx_, ny_;
  real tau_, fx_, fy_;
  Mask solid_;
  // Raw arrays instead of std::vector: memory is not touched at allocation, so the first
  // write (done in parallel with the same row distribution as step()) places each page
  // near the thread that will use it (first-touch policy on NUMA systems).
  std::unique_ptr<real[]> g_, g_new_;
};

}  // namespace lbm
