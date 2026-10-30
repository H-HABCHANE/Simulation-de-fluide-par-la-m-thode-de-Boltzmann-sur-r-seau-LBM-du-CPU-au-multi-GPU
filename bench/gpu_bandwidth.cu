// GPU memory bandwidth, the GPU counterpart of STREAM Copy.
//
// A copy kernel b[i] = a[i] on two large arrays (far larger than the L2 cache), timed with
// CUDA events after a warm-up; bandwidth = 2 * bytes / time (one read, one write per element),
// counted like STREAM Copy. cudaMemcpy device-to-device is measured as a second reference.
// Writes one CSV row (with a header) to the file given as first argument, or to stdout.
//
// Usage: gpu_bandwidth [output.csv]
#include <cstdio>
#include <ctime>
#include <string>

#include "cuda_check.hpp"

namespace {

__global__ void copy_kernel(const float* __restrict__ a, float* __restrict__ b, std::size_t n) {
  // Grid-stride loop: consecutive threads access consecutive elements (coalesced).
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    b[i] = a[i];
}

template <class F>
float best_ms(F&& launch, int reps) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  launch();  // warm-up
  CUDA_CHECK(cudaDeviceSynchronize());
  float best = 1e30f;
  for (int r = 0; r < reps; ++r) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    if (ms < best) best = ms;
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t n = std::size_t(1) << 26;  // 64 Mi floats = 256 MiB per array
  const std::size_t bytes = n * sizeof(float);
  const int reps = 20;

  int device = 0, driver = 0, runtime = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));

  float *a = nullptr, *b = nullptr;
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  CUDA_CHECK(cudaMemset(a, 0, bytes));
  CUDA_CHECK(cudaMemset(b, 0, bytes));

  const int block = 256;
  const int grid = prop.multiProcessorCount * 32;  // enough resident warps to hide latency
  const float kernel_ms = best_ms(
      [&] {
        copy_kernel<<<grid, block>>>(a, b, n);
        CUDA_CHECK_LAUNCH();
      },
      reps);
  const float memcpy_ms =
      best_ms([&] { CUDA_CHECK(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice)); }, reps);

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));

  const double copy_mbps = 2.0 * bytes / (kernel_ms * 1e-3) / 1e6;
  const double memcpy_mbps = 2.0 * bytes / (memcpy_ms * 1e-3) / 1e6;
  auto ver = [](int v) { return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10); };
  char date[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%S", std::localtime(&now));

  FILE* out = argc > 1 ? std::fopen(argv[1], "w") : stdout;
  if (!out) {
    std::perror(argv[1]);
    return 1;
  }
  std::fprintf(out,
               "gpu,copy_MBps,memcpy_MBps,bytes_per_array,reps,ecc,gpu_driver_cuda,"
               "cuda_runtime,cuda_compiled,date\n");
  std::fprintf(out, "\"%s\",%.1f,%.1f,%zu,%d,%d,%s,%s,%s,%s\n", prop.name, copy_mbps,
               memcpy_mbps, bytes, reps, prop.ECCEnabled, ver(driver).c_str(),
               ver(runtime).c_str(), ver(CUDART_VERSION).c_str(), date);
  if (out != stdout) std::fclose(out);
  std::printf("%s: copy kernel %.1f GB/s, cudaMemcpy D2D %.1f GB/s (ECC %s)\n", prop.name,
              copy_mbps / 1e3, memcpy_mbps / 1e3, prop.ECCEnabled ? "on" : "off");
  return 0;
}
