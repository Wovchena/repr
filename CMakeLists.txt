project(repr)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
add_executable(ov ov.cpp)
target_link_libraries(ov openvino::runtime)
