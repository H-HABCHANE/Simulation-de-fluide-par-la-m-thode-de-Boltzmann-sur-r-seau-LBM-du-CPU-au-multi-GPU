# Compiler and flags baked into the binaries, reported by the benchmark mode.
# Target-level options (warnings, -fopenmp) are not included; OpenMP use is reported
# through the thread count.
string(TOUPPER "${CMAKE_BUILD_TYPE}" LBM_BUILD_TYPE_UPPER)
string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${LBM_BUILD_TYPE_UPPER}}" LBM_CXX_FLAGS_ALL)
configure_file(build_info.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/build_info.hpp @ONLY)

add_library(lbm_common STATIC params.cpp geometry.cpp io.cpp)
target_include_directories(lbm_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
if(LBM_REAL_DOUBLE)
  target_compile_definitions(lbm_common PUBLIC LBM_REAL_DOUBLE)
endif()
target_compile_options(lbm_common PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
