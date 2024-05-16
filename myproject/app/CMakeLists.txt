project(myapp)
find_package(mylib REQUIRED CONFIG PATHS "${CMAKE_BINARY_DIR}")
add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE mylib::mylib)
