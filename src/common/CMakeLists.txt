add_library(lbm_common STATIC params.cpp geometry.cpp io.cpp)
target_include_directories(lbm_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(LBM_REAL_DOUBLE)
  target_compile_definitions(lbm_common PUBLIC LBM_REAL_DOUBLE)
endif()
target_compile_options(lbm_common PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
