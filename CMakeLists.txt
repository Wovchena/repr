cmake_minimum_required (VERSION 3.10)
project(repr)
include(FetchContent)

# The flags are from https://github.com/google/benchmark?tab=readme-ov-file#installation
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_ENABLE_LTO ON)
FetchContent_Declare(benchmark GIT_REPOSITORY https://github.com/google/benchmark.git GIT_TAG main)
FetchContent_MakeAvailable(benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
# find_package(OpenCV)
add_executable(repr ov.cpp)
target_link_libraries(repr openvino::runtime benchmark::benchmark)
# target_link_libraries(repr ${OpenCV_LIBRARIES})
