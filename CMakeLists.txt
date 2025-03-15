cmake_minimum_required (VERSION 3.10)
project(repr)

# The flags are from https://github.com/google/benchmark?tab=readme-ov-file#installation
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
add_subdirectory(benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
add_executable(performance_benchmark performance_benchmark.cpp)
target_link_libraries(performance_benchmark benchmark::benchmark openvino::runtime)
# find_package(OpenCV)
# target_link_libraries(performance_benchmark ${OpenCV_LIBRARIES})
