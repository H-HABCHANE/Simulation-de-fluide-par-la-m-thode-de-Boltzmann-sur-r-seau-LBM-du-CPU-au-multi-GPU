cmake_minimum_required(VERSION 3.20)
project(lbm_d2q9 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(LBM_ENABLE_CUDA "Build the CUDA versions (v2, v3)" OFF)
option(LBM_ENABLE_MPI "Build the multi-GPU version (v4, requires CUDA)" OFF)
option(LBM_REAL_DOUBLE "Use double precision for 'real' (default: float)" OFF)

if(LBM_ENABLE_MPI AND NOT LBM_ENABLE_CUDA)
  message(FATAL_ERROR "LBM_ENABLE_MPI=ON requires LBM_ENABLE_CUDA=ON")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
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
add_library(lbm_common INTERFACE)
target_include_directories(lbm_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/common)
if(LBM_REAL_DOUBLE)
  target_compile_definitions(lbm_common INTERFACE LBM_REAL_DOUBLE)
endif()
target_compile_options(lbm_common INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------
add_subdirectory(src/v0_cpu)

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
