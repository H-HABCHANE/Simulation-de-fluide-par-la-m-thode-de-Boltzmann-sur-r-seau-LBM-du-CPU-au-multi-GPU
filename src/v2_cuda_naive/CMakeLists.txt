# v2: naive CUDA port of v0, one thread per cell.
# Default --fmad=false: no fused multiply-add contraction, so the results stay bitwise
# identical to v0 (tested by tests/test_v2_vs_v0.cu). LBM_CUDA_FMAD=ON only exists to measure
# what FMA contraction changes in speed; test_v2_vs_v0 is then expected to fail.
option(LBM_CUDA_FMAD "Allow fused multiply-add contraction in CUDA kernels" OFF)
if(LBM_CUDA_FMAD)
  set(LBM_V2_CUDA_FLAGS --fmad=true)
else()
  set(LBM_V2_CUDA_FLAGS --fmad=false)
endif()

add_library(lbm_v2_lib STATIC solver_v2.cu)
target_include_directories(lbm_v2_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lbm_v2_lib PUBLIC lbm_common)
target_compile_options(lbm_v2_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${LBM_V2_CUDA_FLAGS}>)

# Flags reported in benchmark mode: global CUDA flags, target flags and architectures.
string(TOUPPER "${CMAKE_BUILD_TYPE}" LBM_V2_BT)
string(STRIP "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${LBM_V2_BT}} ${LBM_V2_CUDA_FLAGS} arch=sm_${CMAKE_CUDA_ARCHITECTURES}"
       LBM_V2_REPORTED_FLAGS)

add_executable(lbm_v2 main.cu)
target_link_libraries(lbm_v2 PRIVATE lbm_v2_lib)
target_compile_definitions(lbm_v2 PRIVATE "LBM_EXTRA_FLAGS=\"nvcc: ${LBM_V2_REPORTED_FLAGS}\"")
