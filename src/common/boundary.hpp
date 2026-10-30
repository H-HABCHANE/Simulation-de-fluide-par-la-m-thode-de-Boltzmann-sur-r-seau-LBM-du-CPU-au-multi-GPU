#pragma once

// Inflow / outflow boundaries for a domain that is open in x (periodic in y), shared by all
// versions so that they stay bitwise identical.
//
// Lattice populations are stored shifted (g = f - w). Definitions, in terms of the
// post-streaming populations S (v0's state):
//   inlet  (x = 0):      the 3 populations entering the domain (c_x = +1: i = 1, 5, 8) are set
//                        to the equilibrium of (rho = 1, u = (u_in, 0));
//   outlet (x = nx - 1): the 3 populations entering the domain (c_x = -1: i = 3, 6, 7) are
//                        copied from the column x = nx - 2 (zero gradient along x);
//   populations leaving the domain through x = 0 or x = nx - 1 are discarded.
// Both rules only copy values (or use one precomputed equilibrium), so push (v0) and pull
// (v1, v3) implementations produce exactly the same numbers.

#include <cstddef>

#include "d2q9.hpp"
#include "geometry.hpp"

namespace lbm {

// Shifted equilibrium of (rho0, ux0, uy0) for the 9 directions. Same expressions, in the same
// order, as the init_equilibrium() of every solver, so that an inlet at (1, u_in, 0) and a
// uniform initial state at (1, u_in, 0) are bitwise equal.
inline void equilibrium_state(real rho0, real ux0, real uy0, real* eq) {
  using namespace d2q9;
  const real usq = ux0 * ux0 + uy0 * uy0;
  const real drho0 = rho0 - real(1);
  for (int i = 0; i < Q; ++i) {
    const real cu = cx[i] * ux0 + cy[i] * uy0;
    eq[i] = equilibrium_shifted(w[i], drho0, rho0, cu, usq);
  }
}

// Push scheme (v0, AoS layout g[cell * 9 + i]): applied to the post-streaming buffer.
inline void apply_open_boundaries_push(real* g, const Mask& solid, int nx, int ny,
                                       const real* eq_in) {
  using d2q9::Q;
  for (int y = 0; y < ny; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * nx;
    if (!solid[row])
      for (int i : {1, 5, 8}) g[row * Q + i] = eq_in[i];
    const std::size_t out = row + nx - 1, prev = row + nx - 2;
    if (!solid[out])
      for (int i : {3, 6, 7}) g[out * Q + i] = g[prev * Q + i];
  }
}

// Pull scheme (v1, v3): post-streaming populations S of fluid cell (x, y), gathered from the
// stored post-collision populations R. get(i, cell) returns R_i at `cell`.
// open_x = false: fully periodic domain (Poiseuille).
template <class Get>
void gather_pull(const Get& get, const Mask& solid, int nx, int ny, int x, int y, bool open_x,
                 const real* eq_in, real* out) {
  using namespace d2q9;
  for (int i = 0; i < Q; ++i) {
    if (open_x && x == 0 && cx[i] == 1) {
      out[i] = eq_in[i];
      continue;
    }
    int xc = x;  // cell whose incoming population i is gathered
    if (open_x && x == nx - 1 && cx[i] == -1) xc = nx - 2;  // outlet: copy from x = nx - 2
    const std::size_t cc = static_cast<std::size_t>(y) * nx + xc;
    const int xs = (xc - cx[i] + nx) % nx;
    const int ys = (y - cy[i] + ny) % ny;
    const std::size_t s = static_cast<std::size_t>(ys) * nx + xs;
    out[i] = solid[s] ? get(opp[i], cc) : get(i, s);
  }
}

}  // namespace lbm
