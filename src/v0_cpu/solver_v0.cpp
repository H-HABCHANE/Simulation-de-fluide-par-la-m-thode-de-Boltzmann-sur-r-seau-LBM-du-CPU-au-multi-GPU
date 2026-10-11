#include "solver_v0.hpp"

#include <stdexcept>
#include <utility>

namespace lbm {

using namespace d2q9;

SolverV0::SolverV0(int nx, int ny, real tau, real force_x, real force_y, Mask solid)
    : nx_(nx),
      ny_(ny),
      tau_(tau),
      fx_(force_x),
      fy_(force_y),
      solid_(std::move(solid)),
      g_(static_cast<std::size_t>(nx) * ny * Q, real(0)),
      g_new_(static_cast<std::size_t>(nx) * ny * Q, real(0)) {
  if (solid_.size() != static_cast<std::size_t>(nx) * ny)
    throw std::invalid_argument("solid mask size does not match the grid");
  if (tau <= real(0.5)) throw std::invalid_argument("tau must be > 0.5");
}

void SolverV0::moments(std::size_t c, real& drho, real& ux, real& uy) const {
  const real* gc = &g_[c * Q];
  real m0 = 0, mx = 0, my = 0;
  for (int i = 0; i < Q; ++i) {
    m0 += gc[i];
    mx += cx[i] * gc[i];
    my += cy[i] * gc[i];
  }
  drho = m0;
  const real rho = real(1) + drho;
  // Guo: the physical velocity includes half of the force applied during the step.
  ux = (mx + real(0.5) * fx_) / rho;
  uy = (my + real(0.5) * fy_) / rho;
}

void SolverV0::init_equilibrium(real rho0, real ux0, real uy0) {
  const real usq = ux0 * ux0 + uy0 * uy0;
  const real drho0 = rho0 - real(1);
  for (std::size_t c = 0; c < solid_.size(); ++c) {
    for (int i = 0; i < Q; ++i) {
      const real cu = cx[i] * ux0 + cy[i] * uy0;
      g_[c * Q + i] = solid_[c] ? real(0) : equilibrium_shifted(w[i], drho0, rho0, cu, usq);
    }
  }
  g_new_ = g_;
}

void SolverV0::step() {
  const real omega = real(1) / tau_;
  const real force_prefactor = real(1) - real(0.5) * omega;

  for (int y = 0; y < ny_; ++y) {
    for (int x = 0; x < nx_; ++x) {
      const std::size_t c = cell(x, y);
      if (solid_[c]) continue;

      real drho, ux, uy;
      moments(c, drho, ux, uy);
      const real rho = real(1) + drho;
      const real usq = ux * ux + uy * uy;

      for (int i = 0; i < Q; ++i) {
        // Collision (BGK + Guo forcing). The shift w_i cancels in g - g_eq.
        const real cu = cx[i] * ux + cy[i] * uy;
        const real geq = equilibrium_shifted(w[i], drho, rho, cu, usq);
        const real g_post = g_[c * Q + i] - omega * (g_[c * Q + i] - geq) +
                            force_prefactor * guo_source(w[i], real(cx[i]), real(cy[i]), ux,
                                                         uy, fx_, fy_);

        // Streaming (push) with periodic wrap; bounce back if the target is solid.
        const int xn = (x + cx[i] + nx_) % nx_;
        const int yn = (y + cy[i] + ny_) % ny_;
        const std::size_t cn = cell(xn, yn);
        if (solid_[cn])
          g_new_[c * Q + opp[i]] = g_post;
        else
          g_new_[cn * Q + i] = g_post;
      }
    }
  }
  std::swap(g_, g_new_);
}

void SolverV0::run(int steps) {
  for (int s = 0; s < steps; ++s) step();
}

void SolverV0::macroscopic(std::vector<real>& rho, std::vector<real>& ux,
                           std::vector<real>& uy) const {
  const std::size_t n = solid_.size();
  rho.assign(n, real(0));
  ux.assign(n, real(0));
  uy.assign(n, real(0));
  for (std::size_t c = 0; c < n; ++c) {
    if (solid_[c]) continue;
    real drho;
    moments(c, drho, ux[c], uy[c]);
    rho[c] = real(1) + drho;
  }
}

double SolverV0::total_mass() const {
  // Accumulate the deviation and the rest mass separately to keep the drift visible.
  double drho = 0.0;
  std::size_t fluid = 0;
  for (std::size_t c = 0; c < solid_.size(); ++c) {
    if (solid_[c]) continue;
    ++fluid;
    for (int i = 0; i < Q; ++i) drho += g_[c * Q + i];
  }
  return static_cast<double>(fluid) + drho;
}

}  // namespace lbm
