#pragma once

#include <cstdint>
#include <vector>

#include "d2q9.hpp"

namespace lbm {

// Solid mask, row-major: solid[y * nx + x] = 1 for a solid cell, 0 for fluid.
using Mask = std::vector<std::uint8_t>;

// Plane channel: solid rows y = 0 and y = ny - 1, periodic in x.
// With halfway bounce-back the walls sit at y = 0.5 and y = ny - 1.5,
// so the channel height is H = ny - 2.
Mask make_channel(int nx, int ny);

// Circular cylinder of diameter D centred at (xc, yc), in cell coordinates: a cell is solid
// when its centre lies in the disc (staircase approximation, first order in space).
Mask make_cylinder(int nx, int ny, double xc, double yc, double D);

// Body force giving a Poiseuille profile of peak velocity u_max in a channel of height H
// (rho = 1): u_max = F H^2 / (8 nu).
inline real poiseuille_force(int H, real tau, real u_max) {
  return real(8) * d2q9::viscosity(tau) * u_max / (real(H) * real(H));
}

// Analytical Poiseuille velocity at distance s from the effective wall (rho = 1).
inline double poiseuille_velocity(double s, int H, double nu, double force) {
  return force / (2.0 * nu) * s * (H - s);
}

}  // namespace lbm
