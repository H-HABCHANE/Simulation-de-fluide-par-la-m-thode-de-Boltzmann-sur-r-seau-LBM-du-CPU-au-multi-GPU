#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "d2q9.hpp"
#include "geometry.hpp"

namespace lbm {

// v1: optimised OpenMP CPU solver. Same numerics as v0 (results are bitwise identical), with
// changes measured one at a time against v1_naive (results/v1_steps_*.csv):
//   1. local __restrict__ pointers (no reload of members through `this`);
//   2. periodic neighbours without integer modulo;
//   3. pull streaming with a structure-of-arrays layout, g[i * N + cell], so that the loop
//      over x reads and writes contiguous memory and can be vectorised.
//
// Pull scheme: the stored array R holds post-collision populations. One step gathers the
// populations arriving at each cell (streaming, with bounce-back from solid neighbours),
// collides them and writes the result in place of the cell, in the other buffer.
// v0 stores post-streaming populations S. Streaming is a pure permutation of copies, so with
// R initialised as the inverse permutation of v0's initial state, S = stream(R) at every step
// and each collision sees exactly the same inputs as in v0.
class SolverV1 {
 public:
  SolverV1(int nx, int ny, real tau, real force_x, real force_y, Mask solid);

  void init_equilibrium(real rho0, real ux0, real uy0);
  void step();
  void run(int steps);

  void macroscopic(std::vector<real>& rho, std::vector<real>& ux, std::vector<real>& uy) const;
  double total_mass() const;

  // Post-streaming populations in v0's layout ((y * nx + x) * 9 + i), for comparisons.
  std::vector<real> populations() const;
  int threads() const;

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  const Mask& solid() const { return solid_; }

 private:
  std::size_t cells() const { return static_cast<std::size_t>(nx_) * ny_; }
  // Gathers stream(R) at fluid cell (x, y) into s[0..8].
  void gather(int x, int y, real* s) const;

  int nx_, ny_;
  real tau_, fx_, fy_;
  Mask solid_;
  // Raw arrays: pages are first touched in parallel with the same row distribution as step().
  std::unique_ptr<real[]> g_, g_new_;
};

}  // namespace lbm
