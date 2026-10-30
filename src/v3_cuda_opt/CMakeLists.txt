# v3: optimised CUDA solver (SoA layout, pull streaming), variants selected at run time.
# Same FMA policy as v2 (LBM_CUDA_FMAD, defined in src/v2_cuda_naive): --fmad=false by default,
# so that every variant stays bitwise identical to v0 (tests/test_v3_vs_v0.cu).
if(LBM_CUDA_FMAD)
  set(LBM_V3_CUDA_FLAGS --fmad=true)
else()
  set(LBM_V3_CUDA_FLAGS --fmad=false)
endif()

add_library(lbm_v3_lib STATIC solver_v3.cu)
target_include_directories(lbm_v3_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lbm_v3_lib PUBLIC lbm_common)
target_compile_options(lbm_v3_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${LBM_V3_CUDA_FLAGS}>)

string(TOUPPER "${CMAKE_BUILD_TYPE}" LBM_V3_BT)
string(STRIP "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${LBM_V3_BT}} ${LBM_V3_CUDA_FLAGS} arch=sm_${CMAKE_CUDA_ARCHITECTURES}"
       LBM_V3_REPORTED_FLAGS)

add_executable(lbm_v3 main.cu)
target_link_libraries(lbm_v3 PRIVATE lbm_v3_lib)
target_compile_definitions(lbm_v3 PRIVATE "LBM_EXTRA_FLAGS=\"nvcc: ${LBM_V3_REPORTED_FLAGS}\"")
