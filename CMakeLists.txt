cmake_minimum_required(VERSION 3.20)
project(lbm_d2q9 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(LBM_ENABLE_CUDA "Build the CUDA versions (v2, v3)" OFF)
option(LBM_ENABLE_MPI "Build the multi-GPU version (v4, requires CUDA)" OFF)
option(LBM_REAL_DOUBLE "Use double precision for 'real' (default: float)" OFF)
option(LBM_NATIVE "Optimise for the build machine (-march=native); binaries are not portable" OFF)

if(LBM_ENABLE_MPI AND NOT LBM_ENABLE_CUDA)
  message(FATAL_ERROR "LBM_ENABLE_MPI=ON requires LBM_ENABLE_CUDA=ON")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(LBM_NATIVE)
  # Added to the global flags so that they are recorded in build_info.hpp (benchmark CSV).
  # -ffp-contract=off: no fused multiply-add, so the CPU versions stay bitwise identical to
  # each other. Measured cost: none when memory-bound, about 8% on grids that fit in cache
  # (results/v1_steps_*.csv, steps 4a and 4b).
  string(APPEND CMAKE_CXX_FLAGS " -march=native -ffp-contract=off")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(OpenMP COMPONENTS CXX)

if(LBM_ENABLE_CUDA)
  # Default target: Tesla T4 (Turing, sm_75). Override with -DCMAKE_CUDA_ARCHITECTURES=...
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(LBM_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

# ---------------------------------------------------------------------------
# Common code shared by all versions
# ---------------------------------------------------------------------------
add_subdirectory(src/common)

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------
add_subdirectory(src/v0_cpu)
add_subdirectory(src/v1_omp)
if(LBM_ENABLE_CUDA)
  add_subdirectory(src/v2_cuda_naive)
  add_subdirectory(src/v3_cuda_opt)

  # GPU memory bandwidth (STREAM Copy equivalent), for the GPU ceiling.
  add_executable(gpu_bandwidth bench/gpu_bandwidth.cu)
  target_include_directories(gpu_bandwidth PRIVATE src/common)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

if(LBM_REAL_DOUBLE)
  message(STATUS "LBM: real = double")
else()
  message(STATUS "LBM: real = float")
endif()
message(STATUS "LBM: CUDA=${LBM_ENABLE_CUDA} MPI=${LBM_ENABLE_MPI} OpenMP=${OpenMP_CXX_FOUND}")
