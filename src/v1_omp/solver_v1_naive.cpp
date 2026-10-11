#include "solver_v1_naive.hpp"

#include <omp.h>

#include <stdexcept>
#include <utility>

namespace lbm {

using namespace d2q9;

SolverV1Naive::SolverV1Naive(int nx, int ny, real tau, real force_x, real force_y, Mask solid)
    : nx_(nx),
      ny_(ny),
      tau_(tau),
      fx_(force_x),
      fy_(force_y),
      solid_(std::move(solid)),
      g_(new real[static_cast<std::size_t>(nx) * ny * Q]),
      g_new_(new real[static_cast<std::size_t>(nx) * ny * Q]) {
  if (solid_.size() != static_cast<std::size_t>(nx) * ny)
    throw std::invalid_argument("solid mask size does not match the grid");
  if (tau <= real(0.5)) throw std::invalid_argument("tau must be > 0.5");

  // First touch, with the same static row distribution as step().
#pragma omp parallel for schedule(static)
  for (int y = 0; y < ny_; ++y)
    for (int x = 0; x < nx_; ++x)
      for (int i = 0; i < Q; ++i) {
        g_[cell(x, y) * Q + i] = real(0);
        g_new_[cell(x, y) * Q + i] = real(0);
      }
}

void SolverV1Naive::moments(std::size_t c, real& drho, real& ux, real& uy) const {
  const real* gc = &g_[c * Q];
  real m0 = 0, mx = 0, my = 0;
  for (int i = 0; i < Q; ++i) {
    m0 += gc[i];
    mx += cx[i] * gc[i];
    my += cy[i] * gc[i];
  }
  drho = m0;
  const real rho = real(1) + drho;
  ux = (mx + real(0.5) * fx_) / rho;
  uy = (my + real(0.5) * fy_) / rho;
}

void SolverV1Naive::init_equilibrium(real rho0, real ux0, real uy0) {
  const real usq = ux0 * ux0 + uy0 * uy0;
  const real drho0 = rho0 - real(1);
#pragma omp parallel for schedule(static)
  for (int y = 0; y < ny_; ++y)
    for (int x = 0; x < nx_; ++x) {
      const std::size_t c = cell(x, y);
      for (int i = 0; i < Q; ++i) {
        const real cu = cx[i] * ux0 + cy[i] * uy0;
        const real geq = solid_[c] ? real(0) : equilibrium_shifted(w[i], drho0, rho0, cu, usq);
        g_[c * Q + i] = geq;
        g_new_[c * Q + i] = geq;
      }
    }
}

void SolverV1Naive::step() {
  const real omega = real(1) / tau_;
  const real force_prefactor = real(1) - real(0.5) * omega;

#pragma omp parallel for schedule(static)
  for (int y = 0; y < ny_; ++y) {
    for (int x = 0; x < nx_; ++x) {
      const std::size_t c = cell(x, y);
      if (solid_[c]) continue;

      real drho, ux, uy;
      moments(c, drho, ux, uy);
      const real rho = real(1) + drho;
      const real usq = ux * ux + uy * uy;

      for (int i = 0; i < Q; ++i) {
        const real cu = cx[i] * ux + cy[i] * uy;
        const real geq = equilibrium_shifted(w[i], drho, rho, cu, usq);
        const real g_post = g_[c * Q + i] - omega * (g_[c * Q + i] - geq) +
                            force_prefactor * guo_source(w[i], real(cx[i]), real(cy[i]), ux,
                                                         uy, fx_, fy_);

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

void SolverV1Naive::run(int steps) {
  for (int s = 0; s < steps; ++s) step();
}

void SolverV1Naive::macroscopic(std::vector<real>& rho, std::vector<real>& ux,
                           std::vector<real>& uy) const {
  const std::size_t n = solid_.size();
  rho.assign(n, real(0));
  ux.assign(n, real(0));
  uy.assign(n, real(0));
#pragma omp parallel for schedule(static)
  for (int y = 0; y < ny_; ++y)
    for (int x = 0; x < nx_; ++x) {
      const std::size_t c = cell(x, y);
      if (solid_[c]) continue;
      real drho;
      moments(c, drho, ux[c], uy[c]);
      rho[c] = real(1) + drho;
    }
}

double SolverV1Naive::total_mass() const {
  // Diagnostic only: the parallel reduction order differs from v0 in the last bits.
  double drho = 0.0;
  long fluid = 0;
#pragma omp parallel for schedule(static) reduction(+ : drho, fluid)
  for (int y = 0; y < ny_; ++y)
    for (int x = 0; x < nx_; ++x) {
      const std::size_t c = cell(x, y);
      if (solid_[c]) continue;
      ++fluid;
      for (int i = 0; i < Q; ++i) drho += g_[c * Q + i];
    }
  return static_cast<double>(fluid) + drho;
}

std::vector<real> SolverV1Naive::populations() const { return {g_.get(), g_.get() + size()}; }

int SolverV1Naive::threads() const { return omp_get_max_threads(); }

}  // namespace lbm
