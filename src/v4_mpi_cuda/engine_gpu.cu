// GPU engine of v4: the strip lives in GPU memory (two buffers), the cell update is the same
// strip_update_cell as the CPU engine (compiled for the device), and two CUDA streams let the
// interior rows run while the boundary rows are copied to the host and exchanged.
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "cuda_check.hpp"
#include "engine.hpp"

namespace lbm {

namespace {

// One thread per cell of local rows [first, last]. StripParams is passed by value: it sits in
// the kernel parameter bank and is broadcast to all threads, like __constant__ memory.
__global__ void strip_kernel(const real* __restrict__ g, real* __restrict__ g_new,
                             const std::uint8_t* __restrict__ solid, int first, int last,
                             StripParams P) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = first + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= P.nx || y > last) return;
  strip_update_cell(g, g_new, solid, x, y, P);
}

class GpuEngine final : public StripEngine {
 public:
  GpuEngine(int nx, int ny_local, int local_rank)
      : nx_(nx), nyl_(ny_local), N_(static_cast<std::size_t>(nx) * (ny_local + 2)) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    device_ = local_rank % count;  // one GPU per rank when there are enough of them
    CUDA_CHECK(cudaSetDevice(device_));
    const char* env = std::getenv("LBM_BLOCK");
    if (env) std::sscanf(env, "%dx%d", &bx_, &by_);

    const std::size_t bytes = N_ * d2q9::Q * sizeof(real);
    CUDA_CHECK(cudaMalloc(&g_, bytes));
    CUDA_CHECK(cudaMalloc(&g_new_, bytes));
    CUDA_CHECK(cudaMalloc(&solid_, N_));
    CUDA_CHECK(cudaMemset(g_, 0, bytes));
    CUDA_CHECK(cudaMemset(g_new_, 0, bytes));
    // Page-locked host buffers: required for cudaMemcpyAsync to be truly asynchronous.
    CUDA_CHECK(cudaMallocHost(&halo_, 12 * static_cast<std::size_t>(nx) * sizeof(real)));
    for (auto& s : streams_) CUDA_CHECK(cudaStreamCreate(&s));
  }

  ~GpuEngine() override {
    for (auto& s : streams_) cudaStreamDestroy(s);
    cudaFreeHost(halo_);
    cudaFree(g_);
    cudaFree(g_new_);
    cudaFree(solid_);
  }

  void upload_state(const std::vector<real>& g) override {
    CUDA_CHECK(cudaMemcpy(g_, g.data(), g.size() * sizeof(real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_new_, g.data(), g.size() * sizeof(real), cudaMemcpyHostToDevice));
  }

  void download_state(std::vector<real>& g) const override {
    g.resize(N_ * d2q9::Q);
    CUDA_CHECK(cudaMemcpy(g.data(), g_, g.size() * sizeof(real), cudaMemcpyDeviceToHost));
  }

  void upload_solid(const Mask& solid_strip) override {
    CUDA_CHECK(cudaMemcpy(solid_, solid_strip.data(), N_, cudaMemcpyHostToDevice));
  }

  void set_params(const StripParams& p) override { params_ = p; }

  void compute_rows(int first, int last, Stream s) override {
    if (last < first) return;
    const dim3 block(bx_, by_);
    const dim3 grid((nx_ + bx_ - 1) / bx_, (last - first + 1 + by_ - 1) / by_);
    strip_kernel<<<grid, block, 0, streams_[s]>>>(g_, g_new_, solid_, first, last, params_);
    CUDA_CHECK_LAUNCH();
  }

  void pack_halo() override {
    // Same stream as the boundary rows: the copies start as soon as those rows are done.
    cudaStream_t s = streams_[kBoundary];
    for (int k = 0; k < 3; ++k) {
      CUDA_CHECK(cudaMemcpyAsync(send_up() + k * nx_, row(g_new_, kHaloUp[k], nyl_),
                                 nx_ * sizeof(real), cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaMemcpyAsync(send_down() + k * nx_, row(g_new_, kHaloDown[k], 1),
                                 nx_ * sizeof(real), cudaMemcpyDeviceToHost, s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));  // the interior stream keeps running
  }

  void unpack_halo() override {
    cudaStream_t s = streams_[kBoundary];
    for (int k = 0; k < 3; ++k) {
      CUDA_CHECK(cudaMemcpyAsync(row(g_new_, kHaloUp[k], 0), recv_from_down() + k * nx_,
                                 nx_ * sizeof(real), cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(row(g_new_, kHaloDown[k], nyl_ + 1), recv_from_up() + k * nx_,
                                 nx_ * sizeof(real), cudaMemcpyHostToDevice, s));
    }
  }

  void finish_step() override {
    // Both streams must be done (interior rows and ghost-row uploads) before the next step
    // reads this buffer. Synchronising the streams also reports kernel errors.
    for (auto& s : streams_) CUDA_CHECK(cudaStreamSynchronize(s));
    std::swap(g_, g_new_);
  }

  real* send_up() override { return halo_; }
  real* send_down() override { return halo_ + 3 * nx_; }
  real* recv_from_down() override { return halo_ + 6 * nx_; }
  real* recv_from_up() override { return halo_ + 9 * nx_; }

  std::string backend() const override { return "gpu"; }
  std::string device_name() const override {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_));
    return prop.name;
  }
  std::string device_id() const override {
    char bus[64] = {};
    CUDA_CHECK(cudaDeviceGetPCIBusId(bus, sizeof bus, device_));
    return bus;
  }

 private:
  real* row(real* g, int i, int y) const {
    return g + i * N_ + static_cast<std::size_t>(y) * nx_;
  }

  int nx_, nyl_;
  std::size_t N_;
  int device_ = 0;
  int bx_ = 32, by_ = 8;
  real* g_ = nullptr;
  real* g_new_ = nullptr;
  std::uint8_t* solid_ = nullptr;
  real* halo_ = nullptr;
  cudaStream_t streams_[2] = {};
  StripParams params_{};
};

}  // namespace

std::unique_ptr<StripEngine> make_engine(int nx, int ny_local, int local_rank) {
  return std::make_unique<GpuEngine>(nx, ny_local, local_rank);
}

}  // namespace lbm
