cmake_minimum_required(VERSION 3.6)
project(videocards)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Wall -Wextra -fopenmp")
# set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCL REQUIRED)

add_executable(matrix_mul matrix_mul.c matrix.h print_opencl_error.h)
add_executable(prefix_sum prefix_sum.c prefix.h print_opencl_error.h)

target_link_libraries(matrix_mul OpenCL::OpenCL)
target_link_libraries(prefix_sum OpenCL::OpenCL)
