#include "solver_v2.hpp"

#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "cuda_check.hpp"

namespace lbm {

namespace {

// Lattice constants in constant memory: the d2q9.hpp arrays live in host memory and cannot be
// read by a kernel. All threads of a warp read the same index i at the same time, which the
// constant cache serves with a single broadcast.
__constant__ int c_cx[d2q9::Q];
__constant__ int c_cy[d2q9::Q];
__constant__ int c_opp[d2q9::Q];
__constant__ real c_w[d2q9::Q];

// One thread per cell: same operations, in the same order, as SolverV0::step().
__global__ void step_kernel(const real* __restrict__ g, real* __restrict__ g_new,
                            const std::uint8_t* __restrict__ solid, int nx, int ny, real omega,
                            real force_prefactor, real fx, real fy) {
  using d2q9::Q;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;
  const std::size_t c = static_cast<std::size_t>(y) * nx + x;
  if (solid[c]) return;

  // Moments. Array of structures: thread x reads g[c * 9 + i], so the 32 threads of a warp
  // read addresses 36 bytes apart for each i (not coalesced).
  const real* gc = g + c * Q;
  real m0 = 0, mx = 0, my = 0;
  for (int i = 0; i < Q; ++i) {
    m0 += gc[i];
    mx += c_cx[i] * gc[i];
    my += c_cy[i] * gc[i];
  }
  const real drho = m0;
  const real rho = real(1) + drho;
  const real ux = (mx + real(0.5) * fx) / rho;
  const real uy = (my + real(0.5) * fy) / rho;
  const real usq = ux * ux + uy * uy;

  for (int i = 0; i < Q; ++i) {
    // Collision (BGK + Guo forcing).
    const real cu = c_cx[i] * ux + c_cy[i] * uy;
    const real geq = d2q9::equilibrium_shifted(c_w[i], drho, rho, cu, usq);
    const real g_post = gc[i] - omega * (gc[i] - geq) +
                        force_prefactor * d2q9::guo_source(c_w[i], real(c_cx[i]), real(c_cy[i]),
                                                           ux, uy, fx, fy);

    // Streaming (push) with periodic wrap; bounce back if the target is solid.
    const int xn = (x + c_cx[i] + nx) % nx;
    const int yn = (y + c_cy[i] + ny) % ny;
    const std::size_t cn = static_cast<std::size_t>(yn) * nx + xn;
    if (solid[cn])
      g_new[c * Q + c_opp[i]] = g_post;
    else
      g_new[cn * Q + i] = g_post;
  }
}

void parse_block(int& bx, int& by) {
  const char* env = std::getenv("LBM_BLOCK");
  if (!env) return;
  int x = 0, y = 0;
  if (std::sscanf(env, "%dx%d", &x, &y) != 2 || x <= 0 || y <= 0 || x * y > 1024)
    throw std::invalid_argument("LBM_BLOCK must look like 32x8 (at most 1024 threads)");
  bx = x;
  by = y;
}

}  // namespace

SolverV2::SolverV2(int nx, int ny, real tau, real force_x, real force_y, Mask solid)
    : nx_(nx), ny_(ny), tau_(tau), fx_(force_x), fy_(force_y), solid_(std::move(solid)) {
  using d2q9::Q;
  if (solid_.size() != cells()) throw std::invalid_argument("solid mask size does not match the grid");
  if (tau <= real(0.5)) throw std::invalid_argument("tau must be > 0.5");
  parse_block(block_x_, block_y_);

  CUDA_CHECK(cudaMemcpyToSymbol(c_cx, d2q9::cx, sizeof(d2q9::cx)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_cy, d2q9::cy, sizeof(d2q9::cy)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_opp, d2q9::opp, sizeof(d2q9::opp)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_w, d2q9::w, sizeof(d2q9::w)));

  const std::size_t bytes = cells() * Q * sizeof(real);
  CUDA_CHECK(cudaMalloc(&g_d_, bytes));
  CUDA_CHECK(cudaMalloc(&g_new_d_, bytes));
  CUDA_CHECK(cudaMalloc(&solid_d_, cells()));
  CUDA_CHECK(cudaMemcpy(solid_d_, solid_.data(), cells(), cudaMemcpyHostToDevice));
  // Solid cells are never written by the kernel: both buffers start at zero, as in v0.
  CUDA_CHECK(cudaMemset(g_d_, 0, bytes));
  CUDA_CHECK(cudaMemset(g_new_d_, 0, bytes));
}

SolverV2::~SolverV2() {
  // No CUDA_CHECK here: a destructor must not abort during stack unwinding.
  cudaFree(g_d_);
  cudaFree(g_new_d_);
  cudaFree(solid_d_);
}

void SolverV2::init_equilibrium(real rho0, real ux0, real uy0) {
  // Built on the host exactly as in v0, then copied to both device buffers.
  using namespace d2q9;
  std::vector<real> g(cells() * Q, real(0));
  const real usq = ux0 * ux0 + uy0 * uy0;
  const real drho0 = rho0 - real(1);
  for (std::size_t c = 0; c < cells(); ++c)
    for (int i = 0; i < Q; ++i) {
      const real cu = cx[i] * ux0 + cy[i] * uy0;
      g[c * Q + i] = solid_[c] ? real(0) : equilibrium_shifted(w[i], drho0, rho0, cu, usq);
    }
  const std::size_t bytes = g.size() * sizeof(real);
  CUDA_CHECK(cudaMemcpy(g_d_, g.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(g_new_d_, g.data(), bytes, cudaMemcpyHostToDevice));
}

void SolverV2::step() {
  const real omega = real(1) / tau_;
  const real force_prefactor = real(1) - real(0.5) * omega;
  const dim3 block(block_x_, block_y_);
  const dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);
  step_kernel<<<grid, block>>>(g_d_, g_new_d_, solid_d_, nx_, ny_, omega, force_prefactor, fx_,
                               fy_);
  CUDA_CHECK_LAUNCH();
  std::swap(g_d_, g_new_d_);  // pointer swap on the host, no copy
}

void SolverV2::run(int steps) {
  for (int s = 0; s < steps; ++s) step();
  CUDA_CHECK(cudaDeviceSynchronize());  // reports errors raised while the kernels ran
}

double SolverV2::timed_run(int steps) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int s = 0; s < steps; ++s) step();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return static_cast<double>(ms) * 1e-3;
}

std::vector<real> SolverV2::populations() const {
  std::vector<real> g(cells() * d2q9::Q);
  CUDA_CHECK(cudaMemcpy(g.data(), g_d_, g.size() * sizeof(real), cudaMemcpyDeviceToHost));
  return g;
}

void SolverV2::macroscopic(std::vector<real>& rho, std::vector<real>& ux,
                           std::vector<real>& uy) const {
  // Same formulas and operation order as SolverV0::moments().
  using namespace d2q9;
  const std::vector<real> g = populations();
  const std::size_t n = cells();
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

double SolverV2::total_mass() const {
  const std::vector<real> g = populations();
  double drho = 0.0;
  std::size_t fluid = 0;
  for (std::size_t c = 0; c < cells(); ++c) {
    if (solid_[c]) continue;
    ++fluid;
    for (int i = 0; i < d2q9::Q; ++i) drho += g[c * d2q9::Q + i];
  }
  return static_cast<double>(fluid) + drho;
}

std::string SolverV2::bench_info() const {
  int device = 0, driver = 0, runtime = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  auto ver = [](int v) { return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10); };
  return "\"gpu\": \"" + std::string(prop.name) + "\", \"gpu_driver_cuda\": \"" + ver(driver) +
         "\", \"cuda_runtime\": \"" + ver(runtime) + "\", \"cuda_compiled\": \"" +
         ver(CUDART_VERSION) + "\", \"ecc\": " + std::to_string(prop.ECCEnabled) +
         ", \"block\": \"" + std::to_string(block_x_) + "x" + std::to_string(block_y_) + "\"";
}

}  // namespace lbm
