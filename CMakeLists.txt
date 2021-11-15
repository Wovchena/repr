project(repr)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED COMPONENTS core highgui videoio imgproc imgcodecs)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
add_executable(ov ov.cpp)
target_link_libraries(ov openvino::runtime ${OpenCV_LIBRARIES})

find_package(InferenceEngine REQUIRED)
add_executable(ie ie.cpp)
target_link_libraries(ie IE::inference_engine ${OpenCV_LIBRARIES})
