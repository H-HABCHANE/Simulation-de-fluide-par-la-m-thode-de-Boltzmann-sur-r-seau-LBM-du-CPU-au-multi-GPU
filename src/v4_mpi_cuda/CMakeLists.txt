# v4: domain decomposition in strips of rows, one MPI rank per GPU, halo exchange through
# page-locked host buffers, overlap of the interior computation with the exchange.
# The engine is chosen at build time: GPU (CUDA) or CPU (to test the MPI logic without a GPU).
if(LBM_ENABLE_CUDA)
  set(LBM_V4_ENGINE engine_gpu.cu)
else()
  set(LBM_V4_ENGINE engine_cpu.cpp)
endif()

add_library(lbm_v4_lib STATIC solver_v4.cpp ${LBM_V4_ENGINE})
target_include_directories(lbm_v4_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lbm_v4_lib PUBLIC lbm_common MPI::MPI_CXX)
# Same FMA policy as v2 and v3: bitwise identical to v0 unless LBM_CUDA_FMAD=ON.
# The executables and tests linking this library are plain C++: link the CUDA runtime
# explicitly.
if(LBM_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(lbm_v4_lib PUBLIC CUDA::cudart)
  if(LBM_CUDA_FMAD)
    target_compile_options(lbm_v4_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
  else()
    target_compile_options(lbm_v4_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  endif()
endif()

add_executable(lbm_v4 main.cpp)
target_link_libraries(lbm_v4 PRIVATE lbm_v4_lib)
