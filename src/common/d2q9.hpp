#pragma once

#include "real.hpp"

// Functions marked LBM_HD are callable from both host and CUDA device code.
#if defined(__CUDACC__)
#define LBM_HD __host__ __device__
#else
#define LBM_HD
#endif

namespace lbm::d2q9 {

/* Lattice directions (lattice units: dx = dt = 1)

     6   2   5
      \  |  /
     3 - 0 - 1
      /  |  \
     7   4   8
*/
constexpr int Q = 9;
constexpr int cx[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constexpr int cy[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
constexpr int opp[Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
constexpr real w[Q] = {real(4) / 9,  real(1) / 9,  real(1) / 9,  real(1) / 9, real(1) / 9,
                       real(1) / 36, real(1) / 36, real(1) / 36, real(1) / 36};

// Kinematic viscosity of the BGK model: nu = cs^2 (tau - 1/2), with cs^2 = 1/3.
LBM_HD inline real viscosity(real tau) { return (tau - real(0.5)) / real(3); }

// Populations are stored shifted by their rest value: g_i = f_i - w_i (rest state rho = 1, u = 0).
// The stored numbers are then of the order of the flow perturbation instead of w_i, so float
// rounding acts on the useful part. Moments: rho = 1 + drho with drho = sum_i g_i,
// rho u = sum_i g_i c_i (since sum_i w_i c_i = 0).
//
// Shifted second-order equilibrium, f_eq - w with
// f_eq = w rho [1 + 3 (c.u) + 9/2 (c.u)^2 - 3/2 |u|^2]:
//   g_eq = w [drho + rho (3 (c.u) + 9/2 (c.u)^2 - 3/2 |u|^2)].
LBM_HD inline real equilibrium_shifted(real wi, real drho, real rho, real cu, real usq) {
  return wi * (drho + rho * (real(3) * cu + real(4.5) * cu * cu - real(1.5) * usq));
}

// Guo forcing term without its (1 - 1/(2 tau)) prefactor:
// w [3 (c - u).F + 9 (c.u)(c.F)].
LBM_HD inline real guo_source(real wi, real cxi, real cyi, real ux, real uy, real fx, real fy) {
  const real cu = cxi * ux + cyi * uy;
  const real cf = cxi * fx + cyi * fy;
  return wi * (real(3) * ((cxi - ux) * fx + (cyi - uy) * fy) + real(9) * cu * cf);
}

}  // namespace lbm::d2q9
