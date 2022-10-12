project(repr)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
# find_package(OpenCV)
add_executable(repr ov.cpp)
target_link_libraries(repr openvino::runtime)
# target_link_libraries(repr ${OpenCV_LIBRARIES})
