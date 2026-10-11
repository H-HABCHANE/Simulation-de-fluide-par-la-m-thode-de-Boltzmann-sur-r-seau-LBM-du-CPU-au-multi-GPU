#pragma once

#include <cstddef>
#include <vector>

#include "d2q9.hpp"
#include "geometry.hpp"

namespace lbm {

// v0: sequential CPU reference.
// D2Q9 lattice, BGK collision, Guo body force, halfway bounce-back on solid cells,
// periodic domain edges. Push streaming with two buffers (g -> g_new, then swap).
// Populations are stored shifted, g_i = f_i - w_i (see d2q9.hpp), for float accuracy.
// Memory layout: array of structures, g[(y * nx + x) * 9 + i].
class SolverV0 {
 public:
  SolverV0(int nx, int ny, real tau, real force_x, real force_y, Mask solid);

  // Sets every fluid cell to the equilibrium of (rho0, ux0, uy0).
  void init_equilibrium(real rho0, real ux0, real uy0);

  // One collide + stream step.
  void step();
  void run(int steps);

  // Macroscopic fields; u includes the Guo half-force correction.
  // Solid cells get rho = 0 and u = 0.
  void macroscopic(std::vector<real>& rho, std::vector<real>& ux, std::vector<real>& uy) const;

  // Sum of rho over fluid cells, accumulated in double.
  double total_mass() const;

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  const Mask& solid() const { return solid_; }

 private:
  std::size_t cell(int x, int y) const { return static_cast<std::size_t>(y) * nx_ + x; }
  void moments(std::size_t c, real& drho, real& ux, real& uy) const;

  int nx_, ny_;
  real tau_, fx_, fy_;
  Mask solid_;
  std::vector<real> g_, g_new_;
};

}  // namespace lbm
