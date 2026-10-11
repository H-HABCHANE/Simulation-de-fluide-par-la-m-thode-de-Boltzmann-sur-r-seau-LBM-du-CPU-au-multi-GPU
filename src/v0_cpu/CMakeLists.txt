add_library(lbm_v0_lib STATIC solver_v0.cpp)
target_include_directories(lbm_v0_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lbm_v0_lib PUBLIC lbm_common)

add_executable(lbm_v0 main.cpp)
target_link_libraries(lbm_v0 PRIVATE lbm_v0_lib)
