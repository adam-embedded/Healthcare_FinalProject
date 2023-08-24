cmake_minimum_required(VERSION 3.25)
project(pose_detection_with_boundingbox)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")


set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
        "Directory that contains the TensorFlow project" )
if(NOT TENSORFLOW_SOURCE_DIR)
    get_filename_component(TENSORFLOW_SOURCE_DIR
            "~/developer_tools/tensorflow_src" ABSOLUTE)
endif()

add_subdirectory(
        "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
        "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)


find_package(OpenCV REQUIRED)

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR /home/adam/development/TensorRT-8.6.1.6/)

# Build the TensorRT inference engine library
# TensorRT is found and linked by the tensorrt-cpp-api
add_subdirectory(libs/tensorrt-cpp-api)

# Build the YoloV8 library
add_library(YoloV8_TRT SHARED src/yolov8.cpp)
target_link_libraries(YoloV8_TRT PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(YoloV8_TRT PUBLIC libs/tensorrt-cpp-api/src)



add_executable(pose_detection_with_boundingbox src/main.cpp src/tlite.h src/tlite.cpp)
target_link_libraries(pose_detection_with_boundingbox tensorflow-lite YoloV8_TRT)