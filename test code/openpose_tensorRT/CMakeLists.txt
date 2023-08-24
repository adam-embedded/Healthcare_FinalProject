cmake_minimum_required(VERSION 3.25)
project(openpose_tensorRT)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

#include openCV
find_package( OpenCV REQUIRED )

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR /home/adam/development/TensorRT-8.6.1.6/)

# Build the TensorRT inference engine library
# TensorRT is found and linked by the tensorrt-cpp-api
add_subdirectory(libs/tensorrt-cpp-api)


include_directories(${OpenCV_INCLUDE_DIRS} inc)



add_executable(${PROJECT_NAME} src/main.cpp inc/openpose.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} tensorrt_cpp_api)
target_include_directories(${PROJECT_NAME} PUBLIC libs/tensorrt-cpp-api/src)