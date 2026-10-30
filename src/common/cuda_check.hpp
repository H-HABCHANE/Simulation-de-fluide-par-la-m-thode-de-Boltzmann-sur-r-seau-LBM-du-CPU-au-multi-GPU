#pragma once

// Error checking for the CUDA runtime API.
//
// CUDA_CHECK(call): every runtime call returns a cudaError_t; on failure, print the call,
// the file, the line and the error message, then abort.
// CUDA_CHECK_LAUNCH(): a kernel launch returns nothing, so configuration errors (bad grid or
// block size, too many resources) are only visible through cudaGetLastError() right after
// the launch. Errors raised while the kernel runs are reported by the next synchronising
// call, which is itself checked.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                  \
  do {                                                                                    \
    const cudaError_t lbm_err_ = (call);                                                  \
    if (lbm_err_ != cudaSuccess) {                                                        \
      std::fprintf(stderr, "CUDA error: %s\n  in %s\n  at %s:%d\n",                       \
                   cudaGetErrorString(lbm_err_), #call, __FILE__, __LINE__);              \
      std::abort();                                                                       \
    }                                                                                     \
  } while (0)

#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
