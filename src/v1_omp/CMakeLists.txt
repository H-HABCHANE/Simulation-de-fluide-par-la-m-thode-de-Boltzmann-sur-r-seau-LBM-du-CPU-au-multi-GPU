if(NOT OpenMP_CXX_FOUND)
  message(WARNING "OpenMP not found: v1 is not built")
  return()
endif()

# v1_naive: OpenMP port of v0, kept as the baseline of the optimisation steps.
add_library(lbm_v1_naive_lib STATIC solver_v1_naive.cpp)
target_include_directories(lbm_v1_naive_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lbm_v1_naive_lib PUBLIC lbm_common OpenMP::OpenMP_CXX)

add_executable(lbm_v1_naive main_naive.cpp)
target_link_libraries(lbm_v1_naive PRIVATE lbm_v1_naive_lib)

# v1: optimised OpenMP version.
# -fno-trapping-math lets the compiler compute the collision of every cell unconditionally
# (solid cells are masked afterwards), which the loop vectoriser needs. It does not allow
# reassociation or contraction, so the results stay bitwise identical to v0 (tested).
set(LBM_V1_FLAGS -fno-trapping-math)

add_library(lbm_v1_lib STATIC solver_v1.cpp)
target_include_directories(lbm_v1_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lbm_v1_lib PUBLIC lbm_common OpenMP::OpenMP_CXX)
target_compile_options(lbm_v1_lib PRIVATE ${LBM_V1_FLAGS})

add_executable(lbm_v1 main.cpp)
target_link_libraries(lbm_v1 PRIVATE lbm_v1_lib)
target_compile_definitions(lbm_v1 PRIVATE "LBM_EXTRA_FLAGS=\"${LBM_V1_FLAGS}\"")
