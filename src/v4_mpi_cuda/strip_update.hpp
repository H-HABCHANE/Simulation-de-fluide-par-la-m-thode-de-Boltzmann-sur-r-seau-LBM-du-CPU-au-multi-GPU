#pragma once

// Pull update of one cell of a strip, shared by the CPU and the GPU engines of v4.
//
// A strip holds the rows owned by one MPI rank plus one ghost row below (local row 0) and one
// above (local row ny_local + 1); populations are stored as structure of arrays,
// g[i * N + y * nx + x] with N = nx * (ny_local + 2). The ghost rows make every owned row have
// both y-neighbours locally, so no wrap in y is needed here (periodicity in y is handled by the
// ring of ranks); x is periodic, or open with the inflow / outflow rules of boundary.hpp.
//
// Same floating-point operations, in the same order, as SolverV0::step(): v4 is bitwise
// identical to v0 for any number of ranks.

#include <cstddef>
#include <cstdint>

#include "d2q9.hpp"

namespace lbm {

// Everything a cell update needs besides the arrays. Passed by value to the CUDA kernel, so it
// lives in the kernel parameter (constant) bank and is broadcast to all threads.
struct StripParams {
  int cx[d2q9::Q], cy[d2q9::Q], opp[d2q9::Q];
  real w[d2q9::Q];
  real eq_in[d2q9::Q];  // inlet equilibrium (open domain)
  real omega, force_prefactor, fx, fy;
  int nx, ny_local;
  int open_x;  // 0: periodic in x, 1: inlet at x = 0, outflow at x = nx - 1
};

LBM_HD inline void strip_update_cell(const real* __restrict__ g, real* __restrict__ g_new,
                                     const std::uint8_t* __restrict__ solid, int x, int y,
                                     const StripParams& P) {
  constexpr int Q = d2q9::Q;
  const int nx = P.nx;
  const std::size_t N = static_cast<std::size_t>(nx) * (P.ny_local + 2);
  const std::size_t c = static_cast<std::size_t>(y) * nx + x;
  if (solid[c]) return;

  real gin[Q];
#if defined(__CUDACC__)
#pragma unroll
#endif
  for (int i = 0; i < Q; ++i) {
    if (P.open_x && x == 0 && P.cx[i] == 1) {
      gin[i] = P.eq_in[i];  // inlet
    } else if (P.open_x && x == nx - 1 && P.cx[i] == -1) {
      // Outlet: what the cell x = nx - 2 receives in direction i (zero gradient).
      const std::size_t c2 = c - 1;
      const std::size_t s = static_cast<std::size_t>(y - P.cy[i]) * nx + (nx - 1);
      gin[i] = solid[s] ? g[P.opp[i] * N + c2] : g[i * N + s];
    } else {
      int xs = x - P.cx[i];
      xs = xs < 0 ? nx - 1 : (xs >= nx ? 0 : xs);
      const std::size_t s = static_cast<std::size_t>(y - P.cy[i]) * nx + xs;
      gin[i] = solid[s] ? g[P.opp[i] * N + c] : g[i * N + s];
    }
  }

  real m0 = 0, mx = 0, my = 0;
#if defined(__CUDACC__)
#pragma unroll
#endif
  for (int i = 0; i < Q; ++i) {
    m0 += gin[i];
    mx += P.cx[i] * gin[i];
    my += P.cy[i] * gin[i];
  }
  const real drho = m0;
  const real rho = real(1) + drho;
  const real ux = (mx + real(0.5) * P.fx) / rho;
  const real uy = (my + real(0.5) * P.fy) / rho;
  const real usq = ux * ux + uy * uy;

#if defined(__CUDACC__)
#pragma unroll
#endif
  for (int i = 0; i < Q; ++i) {
    const real cu = P.cx[i] * ux + P.cy[i] * uy;
    const real geq = d2q9::equilibrium_shifted(P.w[i], drho, rho, cu, usq);
    const real g_post =
        gin[i] - P.omega * (gin[i] - geq) +
        P.force_prefactor *
            d2q9::guo_source(P.w[i], real(P.cx[i]), real(P.cy[i]), ux, uy, P.fx, P.fy);
    g_new[i * N + c] = g_post;
  }
}

}  // namespace lbm
