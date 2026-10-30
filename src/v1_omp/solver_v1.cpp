#include "solver_v1.hpp"

#include <omp.h>

#include <stdexcept>
#include <utility>

#include "boundary.hpp"

namespace lbm {

using namespace d2q9;

namespace {

// BGK collision with Guo forcing of the gathered populations gin of cell c, stored in place of
// c. Moments and collision in the same operation order as v0. Solid cells are stored as zero.
inline __attribute__((always_inline)) void collide_store(
    const real* gin, real* __restrict__ gn, const std::uint8_t* __restrict__ solid,
    std::size_t N, std::size_t c, real omega, real force_prefactor, real fx, real fy) {
  real m0 = 0, mx = 0, my = 0;
#pragma GCC unroll 9
  for (int i = 0; i < Q; ++i) {
    m0 += gin[i];
    mx += cx[i] * gin[i];
    my += cy[i] * gin[i];
  }
  const real drho = m0;
  const real rho = real(1) + drho;
  const real ux = (mx + real(0.5) * fx) / rho;
  const real uy = (my + real(0.5) * fy) / rho;
  const real usq = ux * ux + uy * uy;

#pragma GCC unroll 9
  for (int i = 0; i < Q; ++i) {
    const real cu = cx[i] * ux + cy[i] * uy;
    const real geq = equilibrium_shifted(w[i], drho, rho, cu, usq);
    const real g_post = gin[i] - omega * (gin[i] - geq) +
                        force_prefactor * guo_source(w[i], real(cx[i]), real(cy[i]), ux, uy, fx, fy);
    gn[i * N + c] = solid[c] ? real(0) : g_post;
  }
}

// Pull update of one cell c at column x with periodic neighbour columns xm, xp.
// row[k] is the offset of row y - 1 + k (periodic). Solid cells are computed too (their values
// are never read) and stored as zero, so that the loop over x has no branch and vectorises.
inline __attribute__((always_inline)) void update_cell(
    const real* __restrict__ g, real* __restrict__ gn, const std::uint8_t* __restrict__ solid,
    std::size_t N, const std::size_t* row, int xm, int x, int xp, real omega,
    real force_prefactor, real fx, real fy) {
  const std::size_t c = row[1] + x;
  const int col[3] = {xm, x, xp};

  // Streaming (pull): population i arrives from c - c_i, or bounces back from a solid
  // neighbour as the outgoing population opp(i) of c itself.
  real gin[Q];
#pragma GCC unroll 9
  for (int i = 0; i < Q; ++i) {
    // Both loads are always in bounds and done unconditionally: a select between two values
    // vectorises, a conditional load does not.
    const std::size_t s = row[1 - cy[i]] + col[1 - cx[i]];
    const real streamed = g[i * N + s];
    const real bounced = g[opp[i] * N + c];
    gin[i] = solid[s] ? bounced : streamed;
  }
  collide_store(gin, gn, solid, N, c, omega, force_prefactor, fx, fy);
}

// Open domain, edge columns x = 0 and x = nx - 1: gather with the inflow / outflow rules of
// boundary.hpp (scalar code, two columns only), then the same collision.
void update_open_edge(const real* g, real* gn, const Mask& solid, int nx, int ny, int x, int y,
                      const real* eq_in, real omega, real force_prefactor, real fx, real fy) {
  const std::size_t N = static_cast<std::size_t>(nx) * ny;
  real gin[Q];
  gather_pull([&](int i, std::size_t cell) { return g[i * N + cell]; }, solid, nx, ny, x, y,
              true, eq_in, gin);
  collide_store(gin, gn, solid.data(), N, static_cast<std::size_t>(y) * nx + x, omega,
                force_prefactor, fx, fy);
}

}  // namespace

SolverV1::SolverV1(int nx, int ny, real tau, real force_x, real force_y, Mask solid)
    : nx_(nx),
      ny_(ny),
      tau_(tau),
      fx_(force_x),
      fy_(force_y),
      solid_(std::move(solid)),
      g_(new real[static_cast<std::size_t>(nx) * ny * Q]),
      g_new_(new real[static_cast<std::size_t>(nx) * ny * Q]) {
  if (solid_.size() != cells()) throw std::invalid_argument("solid mask size does not match the grid");
  if (tau <= real(0.5)) throw std::invalid_argument("tau must be > 0.5");

  // First touch, with the same static row distribution as step().
  const std::size_t N = cells();
#pragma omp parallel for schedule(static)
  for (int y = 0; y < ny_; ++y)
    for (int i = 0; i < Q; ++i)
      for (int x = 0; x < nx_; ++x) {
        const std::size_t k = i * N + static_cast<std::size_t>(y) * nx_ + x;
        g_[k] = real(0);
        g_new_[k] = real(0);
      }
}

void SolverV1::init_equilibrium(real rho0, real ux0, real uy0) {
  // v0's initial state S0 is the same equilibrium in every fluid cell. R0 is its preimage by
  // streaming: the slot (c, i) of R0 is sent by streaming to (c + c_i, i), or to (c, opp(i))
  // if c + c_i is solid, so it takes the value S0 has there.
  const real usq = ux0 * ux0 + uy0 * uy0;
  const real drho0 = rho0 - real(1);
  real eq[Q];
  for (int i = 0; i < Q; ++i)
    eq[i] = equilibrium_shifted(w[i], drho0, rho0, cx[i] * ux0 + cy[i] * uy0, usq);

  const std::size_t N = cells();
#pragma omp parallel for schedule(static)
  for (int y = 0; y < ny_; ++y)
    for (int x = 0; x < nx_; ++x) {
      const std::size_t c = static_cast<std::size_t>(y) * nx_ + x;
      for (int i = 0; i < Q; ++i) {
        // Open domain: a slot whose population leaves through x = 0 or x = nx - 1 is never
        // read; it gets eq[i] like the others.
        const bool leaves = open_x_ && (x + cx[i] < 0 || x + cx[i] >= nx_);
        const int xn = (x + cx[i] + nx_) % nx_;
        const int yn = (y + cy[i] + ny_) % ny_;
        const std::size_t cn = static_cast<std::size_t>(yn) * nx_ + xn;
        const real v = solid_[c] ? real(0) : (!leaves && solid_[cn] ? eq[opp[i]] : eq[i]);
        g_[i * N + c] = v;
        g_new_[i * N + c] = v;
      }
    }
}

void SolverV1::step() {
  const real omega = real(1) / tau_;
  const real force_prefactor = real(1) - real(0.5) * omega;

  // Step 1: local copies of the members, __restrict__ pointers.
  const int nx = nx_, ny = ny_;
  const real fx = fx_, fy = fy_;
  const std::size_t N = cells();
  const real* __restrict__ g = g_.get();
  real* __restrict__ gn = g_new_.get();
  const std::uint8_t* __restrict__ solid = solid_.data();
  const bool open = open_x_;
  const real* eq_in = eq_in_;
  const Mask& mask = solid_;

#pragma omp parallel for schedule(static)
  for (int y = 0; y < ny; ++y) {
    // Step 2: periodic rows computed once per row, no modulo.
    const std::size_t row[3] = {static_cast<std::size_t>(y == 0 ? ny - 1 : y - 1) * nx,
                                static_cast<std::size_t>(y) * nx,
                                static_cast<std::size_t>(y == ny - 1 ? 0 : y + 1) * nx};
    // Step 3: the wrapping columns x = 0 and x = nx - 1 are done apart, so the interior loop
    // has unit-stride accesses only and no branch.
    // Open domain: inflow / outflow rules on these two columns (boundary.hpp).
    if (open)
      update_open_edge(g, gn, mask, nx, ny, 0, y, eq_in, omega, force_prefactor, fx, fy);
    else
      update_cell(g, gn, solid, N, row, nx - 1, 0, nx > 1 ? 1 : 0, omega, force_prefactor, fx, fy);
    // g (read) and gn (written) are distinct buffers. The __restrict__ information is lost when
    // OpenMP outlines the loop body, so the absence of dependence is restated here.
#pragma GCC ivdep
    for (int x = 1; x < nx - 1; ++x)
      update_cell(g, gn, solid, N, row, x - 1, x, x + 1, omega, force_prefactor, fx, fy);
    if (open && nx > 1)
      update_open_edge(g, gn, mask, nx, ny, nx - 1, y, eq_in, omega, force_prefactor, fx, fy);
    else if (nx > 1)
      update_cell(g, gn, solid, N, row, nx - 2, nx - 1, 0, omega, force_prefactor, fx, fy);
  }
  std::swap(g_, g_new_);
}

void SolverV1::run(int steps) {
  for (int s = 0; s < steps; ++s) step();
}

void SolverV1::gather(int x, int y, real* s) const {
  const std::size_t N = cells();
  gather_pull([&](int i, std::size_t cell) { return g_[i * N + cell]; }, solid_, nx_, ny_, x, y,
              open_x_, eq_in_, s);
}

void SolverV1::set_inflow_outflow(real u_in) {
  open_x_ = true;
  equilibrium_state(real(1), u_in, real(0), eq_in_);
}

void SolverV1::populations_at(int x, int y, real* out) const { gather(x, y, out); }

void SolverV1::macroscopic(std::vector<real>& rho, std::vector<real>& ux,
                           std::vector<real>& uy) const {
  const std::size_t n = cells();
  rho.assign(n, real(0));
  ux.assign(n, real(0));
  uy.assign(n, real(0));
#pragma omp parallel for schedule(static)
  for (int y = 0; y < ny_; ++y)
    for (int x = 0; x < nx_; ++x) {
      const std::size_t c = static_cast<std::size_t>(y) * nx_ + x;
      if (solid_[c]) continue;
      real s[Q];
      gather(x, y, s);
      real m0 = 0, mx = 0, my = 0;
      for (int i = 0; i < Q; ++i) {
        m0 += s[i];
        mx += cx[i] * s[i];
        my += cy[i] * s[i];
      }
      const real r = real(1) + m0;
      rho[c] = r;
      ux[c] = (mx + real(0.5) * fx_) / r;
      uy[c] = (my + real(0.5) * fy_) / r;
    }
}

double SolverV1::total_mass() const {
  // Streaming permutes the fluid slots, so the sum over R equals the sum over stream(R).
  // Diagnostic only: the summation order differs from v0 in the last bits.
  const std::size_t N = cells();
  double drho = 0.0;
  long fluid = 0;
#pragma omp parallel for schedule(static) reduction(+ : drho, fluid)
  for (int y = 0; y < ny_; ++y)
    for (int x = 0; x < nx_; ++x) {
      const std::size_t c = static_cast<std::size_t>(y) * nx_ + x;
      if (solid_[c]) continue;
      ++fluid;
      for (int i = 0; i < Q; ++i) drho += g_[i * N + c];
    }
  return static_cast<double>(fluid) + drho;
}

std::vector<real> SolverV1::populations() const {
  std::vector<real> out(cells() * Q, real(0));
  for (int y = 0; y < ny_; ++y)
    for (int x = 0; x < nx_; ++x) {
      const std::size_t c = static_cast<std::size_t>(y) * nx_ + x;
      if (!solid_[c]) gather(x, y, &out[c * Q]);
    }
  return out;
}

int SolverV1::threads() const { return omp_get_max_threads(); }

}  // namespace lbm
