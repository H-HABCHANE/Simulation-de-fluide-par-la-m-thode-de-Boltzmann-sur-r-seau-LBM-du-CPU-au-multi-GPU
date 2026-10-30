#include "solver_v3.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "cuda_check.hpp"

namespace lbm {

namespace {

__constant__ int c_cx[d2q9::Q];
__constant__ int c_cy[d2q9::Q];
__constant__ int c_opp[d2q9::Q];
__constant__ real c_w[d2q9::Q];

// Periodic neighbour (x + dx, y + dy) with |dx|, |dy| <= 1.
// kNoModulo: one comparison per coordinate instead of an integer division.
template <bool kNoModulo>
__device__ __forceinline__ std::size_t neighbour(int x, int y, int dx, int dy, int nx, int ny) {
  int xn = x + dx, yn = y + dy;
  if constexpr (kNoModulo) {
    xn = xn < 0 ? nx - 1 : (xn >= nx ? 0 : xn);
    yn = yn < 0 ? ny - 1 : (yn >= ny ? 0 : yn);
  } else {
    xn = (xn + nx) % nx;
    yn = (yn + ny) % ny;
  }
  return static_cast<std::size_t>(yn) * nx + xn;
}

// One thread per cell, SoA layout g[i * N + cell]. Same floating-point operations, in the
// same order, as SolverV0::step().
// __launch_bounds__(1024): caps the kernel at 65536 / 1024 = 64 registers per thread, so that
// any block shape up to 1024 threads reaches full occupancy. The unrolled loops keep the 9
// populations in registers: 64 registers without spill in float; in double the pull variants
// spill 24 bytes per thread to the stack (cuobjdump -res-usage).
template <bool kPull, bool kNoModulo>
__global__ void __launch_bounds__(1024)
    step_kernel(const real* __restrict__ g, real* __restrict__ g_new,
                const std::uint8_t* __restrict__ solid, int nx, int ny, real omega,
                real force_prefactor, real fx, real fy) {
  using d2q9::Q;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;
  const std::size_t N = static_cast<std::size_t>(nx) * ny;
  const std::size_t c = static_cast<std::size_t>(y) * nx + x;
  if (solid[c]) return;

  // Load the 9 populations of the cell. SoA: for a given i, the 32 threads of a warp read 32
  // consecutive floats (one 128-byte line), shifted by at most one cell in pull mode.
  real gin[Q];
#pragma unroll
  for (int i = 0; i < Q; ++i) {
    if constexpr (kPull) {
      // Population i arrives from c - c_i, or bounces back from a solid neighbour as the
      // outgoing population opp(i) of c itself.
      const std::size_t s = neighbour<kNoModulo>(x, y, -c_cx[i], -c_cy[i], nx, ny);
      gin[i] = solid[s] ? g[c_opp[i] * N + c] : g[i * N + s];
    } else {
      gin[i] = g[i * N + c];
    }
  }

  real m0 = 0, mx = 0, my = 0;
#pragma unroll
  for (int i = 0; i < Q; ++i) {
    m0 += gin[i];
    mx += c_cx[i] * gin[i];
    my += c_cy[i] * gin[i];
  }
  const real drho = m0;
  const real rho = real(1) + drho;
  const real ux = (mx + real(0.5) * fx) / rho;
  const real uy = (my + real(0.5) * fy) / rho;
  const real usq = ux * ux + uy * uy;

#pragma unroll
  for (int i = 0; i < Q; ++i) {
    const real cu = c_cx[i] * ux + c_cy[i] * uy;
    const real geq = d2q9::equilibrium_shifted(c_w[i], drho, rho, cu, usq);
    const real g_post = gin[i] - omega * (gin[i] - geq) +
                        force_prefactor * d2q9::guo_source(c_w[i], real(c_cx[i]), real(c_cy[i]),
                                                           ux, uy, fx, fy);
    if constexpr (kPull) {
      g_new[i * N + c] = g_post;  // own cell: coalesced write
    } else {
      const std::size_t cn = neighbour<kNoModulo>(x, y, c_cx[i], c_cy[i], nx, ny);
      if (solid[cn])
        g_new[c_opp[i] * N + c] = g_post;
      else
        g_new[i * N + cn] = g_post;
    }
  }
}

SolverV3::Variant parse_variant() {
  const char* env = std::getenv("LBM_V3_VARIANT");
  if (!env || std::strcmp(env, "soa_pull_opt") == 0) return SolverV3::Variant::SoaPullOpt;
  if (std::strcmp(env, "soa_pull") == 0) return SolverV3::Variant::SoaPull;
  if (std::strcmp(env, "soa_push") == 0) return SolverV3::Variant::SoaPush;
  throw std::invalid_argument("LBM_V3_VARIANT must be soa_push, soa_pull or soa_pull_opt");
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

const char* SolverV3::variant_name(Variant v) {
  switch (v) {
    case Variant::SoaPush: return "soa_push";
    case Variant::SoaPull: return "soa_pull";
    case Variant::SoaPullOpt: return "soa_pull_opt";
  }
  return "?";
}

SolverV3::SolverV3(int nx, int ny, real tau, real force_x, real force_y, Mask solid)
    : nx_(nx), ny_(ny), tau_(tau), fx_(force_x), fy_(force_y), solid_(std::move(solid)) {
  using d2q9::Q;
  if (solid_.size() != cells()) throw std::invalid_argument("solid mask size does not match the grid");
  if (tau <= real(0.5)) throw std::invalid_argument("tau must be > 0.5");
  variant_ = parse_variant();
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
  CUDA_CHECK(cudaMemset(g_d_, 0, bytes));
  CUDA_CHECK(cudaMemset(g_new_d_, 0, bytes));
}

SolverV3::~SolverV3() {
  cudaFree(g_d_);
  cudaFree(g_new_d_);
  cudaFree(solid_d_);
}

void SolverV3::init_equilibrium(real rho0, real ux0, real uy0) {
  using namespace d2q9;
  const std::size_t N = cells();
  const real usq = ux0 * ux0 + uy0 * uy0;
  const real drho0 = rho0 - real(1);
  real eq[Q];
  for (int i = 0; i < Q; ++i)
    eq[i] = equilibrium_shifted(w[i], drho0, rho0, cx[i] * ux0 + cy[i] * uy0, usq);

  std::vector<real> g(N * Q, real(0));
  for (int y = 0; y < ny_; ++y)
    for (int x = 0; x < nx_; ++x) {
      const std::size_t c = static_cast<std::size_t>(y) * nx_ + x;
      if (solid_[c]) continue;
      for (int i = 0; i < Q; ++i) {
        if (pull()) {
          // Preimage by streaming: slot (c, i) is sent to (c + c_i, i), or to (c, opp(i)) if
          // c + c_i is solid; the equilibrium is the same in every fluid cell.
          const int xn = (x + cx[i] + nx_) % nx_;
          const int yn = (y + cy[i] + ny_) % ny_;
          g[i * N + c] = solid_[static_cast<std::size_t>(yn) * nx_ + xn] ? eq[opp[i]] : eq[i];
        } else {
          g[i * N + c] = eq[i];
        }
      }
    }
  const std::size_t bytes = g.size() * sizeof(real);
  CUDA_CHECK(cudaMemcpy(g_d_, g.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(g_new_d_, g.data(), bytes, cudaMemcpyHostToDevice));
}

void SolverV3::step() {
  const real omega = real(1) / tau_;
  const real force_prefactor = real(1) - real(0.5) * omega;
  const dim3 block(block_x_, block_y_);
  const dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);
  switch (variant_) {
    case Variant::SoaPush:
      step_kernel<false, false><<<grid, block>>>(g_d_, g_new_d_, solid_d_, nx_, ny_, omega,
                                                 force_prefactor, fx_, fy_);
      break;
    case Variant::SoaPull:
      step_kernel<true, false><<<grid, block>>>(g_d_, g_new_d_, solid_d_, nx_, ny_, omega,
                                                force_prefactor, fx_, fy_);
      break;
    case Variant::SoaPullOpt:
      step_kernel<true, true><<<grid, block>>>(g_d_, g_new_d_, solid_d_, nx_, ny_, omega,
                                               force_prefactor, fx_, fy_);
      break;
  }
  CUDA_CHECK_LAUNCH();
  std::swap(g_d_, g_new_d_);
}

void SolverV3::run(int steps) {
  for (int s = 0; s < steps; ++s) step();
  CUDA_CHECK(cudaDeviceSynchronize());
}

double SolverV3::timed_run(int steps) {
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

std::vector<real> SolverV3::populations() const {
  using namespace d2q9;
  const std::size_t N = cells();
  std::vector<real> g(N * Q);
  CUDA_CHECK(cudaMemcpy(g.data(), g_d_, g.size() * sizeof(real), cudaMemcpyDeviceToHost));

  // Convert to v0's layout; in pull mode, gather (stream) the stored post-collision values.
  std::vector<real> out(N * Q, real(0));
  for (int y = 0; y < ny_; ++y)
    for (int x = 0; x < nx_; ++x) {
      const std::size_t c = static_cast<std::size_t>(y) * nx_ + x;
      if (solid_[c]) continue;
      for (int i = 0; i < Q; ++i) {
        if (pull()) {
          const int xs = (x - cx[i] + nx_) % nx_;
          const int ys = (y - cy[i] + ny_) % ny_;
          const std::size_t s = static_cast<std::size_t>(ys) * nx_ + xs;
          out[c * Q + i] = solid_[s] ? g[opp[i] * N + c] : g[i * N + s];
        } else {
          out[c * Q + i] = g[i * N + c];
        }
      }
    }
  return out;
}

void SolverV3::macroscopic(std::vector<real>& rho, std::vector<real>& ux,
                           std::vector<real>& uy) const {
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

double SolverV3::total_mass() const {
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

std::string SolverV3::bench_info() const {
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
         ", \"block\": \"" + std::to_string(block_x_) + "x" + std::to_string(block_y_) +
         "\", \"variant\": \"" + variant_name(variant_) + "\"";
}

}  // namespace lbm
