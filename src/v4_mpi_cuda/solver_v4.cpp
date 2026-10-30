#include "solver_v4.hpp"

#include <mpi.h>

#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <utility>

#include "boundary.hpp"

namespace lbm {

using namespace d2q9;

namespace {

MPI_Datatype mpi_real() { return sizeof(real) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT; }

int local_rank() {
  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int r = 0;
  MPI_Comm_rank(node, &r);
  MPI_Comm_free(&node);
  return r;
}

}  // namespace

SolverV4::SolverV4(int nx, int ny, real tau, real force_x, real force_y, Mask solid)
    : nx_(nx), ny_(ny), tau_(tau), fx_(force_x), fy_(force_y), solid_(std::move(solid)) {
  if (solid_.size() != static_cast<std::size_t>(nx) * ny)
    throw std::invalid_argument("solid mask size does not match the grid");
  if (tau <= real(0.5)) throw std::invalid_argument("tau must be > 0.5");
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  strip_ = make_strip(ny, rank, size);
  const char* ov = std::getenv("LBM_V4_OVERLAP");
  overlap_ = !(ov && std::strcmp(ov, "0") == 0);

  // Solid mask of the strip, ghost rows included: every rank knows the global geometry, so
  // the ghost rows are read from it directly (no exchange needed).
  solid_strip_.assign(strip_cells(), 0);
  for (int ly = 0; ly < strip_.ny_local + 2; ++ly)
    for (int x = 0; x < nx_; ++x)
      solid_strip_[static_cast<std::size_t>(ly) * nx_ + x] =
          solid_[static_cast<std::size_t>(global_row(ly)) * nx_ + x];

  for (int i = 0; i < Q; ++i) {
    params_.cx[i] = cx[i];
    params_.cy[i] = cy[i];
    params_.opp[i] = opp[i];
    params_.w[i] = w[i];
  }
  params_.omega = real(1) / tau_;
  params_.force_prefactor = real(1) - real(0.5) * params_.omega;
  params_.fx = fx_;
  params_.fy = fy_;
  params_.nx = nx_;
  params_.ny_local = strip_.ny_local;
  params_.open_x = 0;

  engine_ = make_engine(nx_, strip_.ny_local, local_rank());
  engine_->upload_solid(solid_strip_);
  engine_->set_params(params_);
}

SolverV4::~SolverV4() = default;

void SolverV4::set_inflow_outflow(real u_in) {
  open_x_ = true;
  equilibrium_state(real(1), u_in, real(0), eq_in_);
  params_.open_x = 1;
  for (int i = 0; i < Q; ++i) params_.eq_in[i] = eq_in_[i];
  engine_->set_params(params_);
}

void SolverV4::init_equilibrium(real rho0, real ux0, real uy0) {
  // Same preimage-by-streaming initial state as v1 and v3, built for the owned rows and the
  // ghost rows from global coordinates, so the ghost rows hold the neighbours' values.
  real eq[Q];
  equilibrium_state(rho0, ux0, uy0, eq);
  const std::size_t N = strip_cells();
  std::vector<real> g(N * Q, real(0));
  for (int ly = 0; ly < strip_.ny_local + 2; ++ly) {
    const int y = global_row(ly);
    for (int x = 0; x < nx_; ++x) {
      if (solid_[static_cast<std::size_t>(y) * nx_ + x]) continue;
      const std::size_t c = static_cast<std::size_t>(ly) * nx_ + x;
      for (int i = 0; i < Q; ++i) {
        const bool leaves = open_x_ && (x + cx[i] < 0 || x + cx[i] >= nx_);
        const int xn = (x + cx[i] + nx_) % nx_;
        const int yn = (y + cy[i] + ny_) % ny_;
        const bool dest_solid = solid_[static_cast<std::size_t>(yn) * nx_ + xn];
        g[i * N + c] = !leaves && dest_solid ? eq[opp[i]] : eq[i];
      }
    }
  }
  engine_->upload_state(g);
}

void SolverV4::exchange_halo() {
  // Tag 0: populations going up (to the rank above), tag 1: going down. With 1 or 2 ranks the
  // neighbour above and below is the same rank, possibly this one: tags keep them apart.
  const int n = 3 * nx_;
  MPI_Request req[4];
  MPI_Irecv(engine_->recv_from_down(), n, mpi_real(), strip_.down, 0, MPI_COMM_WORLD, &req[0]);
  MPI_Irecv(engine_->recv_from_up(), n, mpi_real(), strip_.up, 1, MPI_COMM_WORLD, &req[1]);
  MPI_Isend(engine_->send_up(), n, mpi_real(), strip_.up, 0, MPI_COMM_WORLD, &req[2]);
  MPI_Isend(engine_->send_down(), n, mpi_real(), strip_.down, 1, MPI_COMM_WORLD, &req[3]);
  MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
}

void SolverV4::step() {
  const int nyl = strip_.ny_local;
  if (overlap_) {
    engine_->compute_rows(1, 1, StripEngine::kBoundary);
    engine_->compute_rows(nyl, nyl, StripEngine::kBoundary);
    if (nyl > 2) engine_->compute_rows(2, nyl - 1, StripEngine::kInterior);
  } else {
    engine_->compute_rows(1, nyl, StripEngine::kBoundary);
  }
  engine_->pack_halo();  // waits for the boundary rows only
  exchange_halo();       // the interior rows are still running on the GPU
  engine_->unpack_halo();
  engine_->finish_step();
}

void SolverV4::run(int steps) {
  for (int s = 0; s < steps; ++s) step();
}

double SolverV4::timed_run(int steps) {
  // Each step ends with a full synchronisation of the engine, so the host clock measures the
  // GPU work; the slowest rank sets the time.
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  run(steps);
  double t = MPI_Wtime() - t0, t_max = 0.0;
  MPI_Allreduce(&t, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return t_max;
}

std::vector<real> SolverV4::local_populations() const {
  const std::size_t N = strip_cells();
  std::vector<real> g;
  engine_->download_state(g);
  std::vector<real> out(static_cast<std::size_t>(nx_) * strip_.ny_local * Q, real(0));
  for (int ly = 1; ly <= strip_.ny_local; ++ly)
    for (int x = 0; x < nx_; ++x) {
      if (solid_strip_[static_cast<std::size_t>(ly) * nx_ + x]) continue;
      const std::size_t o = (static_cast<std::size_t>(ly - 1) * nx_ + x) * Q;
      // Ghost rows make y wrap unnecessary: gather_pull's periodic y never triggers here.
      gather_pull([&](int i, std::size_t cell) { return g[i * N + cell]; }, solid_strip_, nx_,
                  strip_.ny_local + 2, x, ly, open_x_, eq_in_, &out[o]);
    }
  return out;
}

std::vector<real> SolverV4::populations() const {
  const std::vector<real> local = local_populations();
  std::vector<int> counts(strip_.size), displs(strip_.size);
  for (int r = 0; r < strip_.size; ++r) {
    const Strip s = make_strip(ny_, r, strip_.size);
    counts[r] = s.ny_local * nx_ * Q;
    displs[r] = s.y0 * nx_ * Q;
  }
  std::vector<real> global(static_cast<std::size_t>(nx_) * ny_ * Q);
  MPI_Allgatherv(local.data(), static_cast<int>(local.size()), mpi_real(), global.data(),
                 counts.data(), displs.data(), mpi_real(), MPI_COMM_WORLD);
  return global;
}

void SolverV4::populations_at(int x, int y, real* out) const {
  int owner = 0;
  for (int r = 0; r < strip_.size; ++r) {
    const Strip s = make_strip(ny_, r, strip_.size);
    if (y >= s.y0 && y < s.y0 + s.ny_local) owner = r;
  }
  if (strip_.rank == owner) {
    const std::vector<real> local = local_populations();
    const std::size_t o = (static_cast<std::size_t>(y - strip_.y0) * nx_ + x) * Q;
    for (int i = 0; i < Q; ++i) out[i] = local[o + i];
  }
  MPI_Bcast(out, Q, mpi_real(), owner, MPI_COMM_WORLD);
}

void SolverV4::macroscopic(std::vector<real>& rho, std::vector<real>& ux,
                           std::vector<real>& uy) const {
  // Same formulas and operation order as SolverV0::moments().
  const std::vector<real> g = populations();
  const std::size_t n = static_cast<std::size_t>(nx_) * ny_;
  rho.assign(n, real(0));
  ux.assign(n, real(0));
  uy.assign(n, real(0));
  for (std::size_t c = 0; c < n; ++c) {
    if (solid_[c]) continue;
    real m0 = 0, mx = 0, my = 0;
    for (int i = 0; i < Q; ++i) {
      m0 += g[c * Q + i];
      mx += cx[i] * g[c * Q + i];
      my += cy[i] * g[c * Q + i];
    }
    const real r = real(1) + m0;
    rho[c] = r;
    ux[c] = (mx + real(0.5) * fx_) / r;
    uy[c] = (my + real(0.5) * fy_) / r;
  }
}

double SolverV4::total_mass() const {
  const std::vector<real> g = populations();
  double drho = 0.0;
  std::size_t fluid = 0;
  for (std::size_t c = 0; c < solid_.size(); ++c) {
    if (solid_[c]) continue;
    ++fluid;
    for (int i = 0; i < Q; ++i) drho += g[c * Q + i];
  }
  return static_cast<double>(fluid) + drho;
}

std::string SolverV4::bench_info() const {
  // Number of distinct devices: rank-to-GPU sharing must be visible in the results, since a
  // scaling measurement is only meaningful with one GPU per rank.
  char id[64] = {};
  std::strncpy(id, engine_->device_id().c_str(), sizeof id - 1);
  std::vector<char> all(64 * static_cast<std::size_t>(strip_.size));
  MPI_Allgather(id, 64, MPI_CHAR, all.data(), 64, MPI_CHAR, MPI_COMM_WORLD);
  std::set<std::string> ids;
  for (int r = 0; r < strip_.size; ++r) ids.insert(std::string(&all[64 * r]));
  std::string info = "\"backend\": \"" + engine_->backend() + "\", \"ranks\": " +
                     std::to_string(strip_.size) + ", \"devices\": " +
                     std::to_string(engine_->backend() == "gpu" ? ids.size() : 0) +
                     ", \"overlap\": " + (overlap_ ? "1" : "0");
  if (!engine_->device_name().empty()) info += ", \"gpu\": \"" + engine_->device_name() + "\"";
  return info;
}

}  // namespace lbm
