cmake_minimum_required(VERSION 3.10)
project(tensorrt_cpp_api)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

## Include tensorRT
#set(TENSORRT_SOURCE_DIR "" CACHE PATH "directory that contains TensorRT")
#if (NOT TENSORRT_SOURCE_DIR)
#    set(TensorRT_DIR /home/adam/development/TensorRT-8.6.1.6)
#    get_filename_component(TENSORRT_SOURCE_DIR ${TensorRT_DIR} ABSOLUTE)
#endif()
#
#set(TENSORRT_INCLUDE_DIR ${TENSORRT_SOURCE_DIR}/include)
#set(TENSORRT_LIB_DIR ${TENSORRT_SOURCE_DIR}/lib)
#
#include_directories(${TENSORRT_INCLUDE_DIR})
#link_directories(${TENSORRT_INCLUDE_DIR})
#
#include_directories(${TENSORRT_LIB_DIR})
#link_directories(${TENSORRT_LIB_DIR})


#list(APPEND PLUGINS "nvinfer")
#list(APPEND PLUGINS "nvonnxparser")
#list(APPEND PLUGINS "nvparsers")
#
#foreach(libName ${PLUGINS})
#    find_library(${libName}_lib NAMES ${libName} ${TensorRT_DIR} PATH_SUFFIXES lib)
#    list(APPEND PLUGIN_LIBS "${${libName}_lib}")
#endforeach()


# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_library(tensorrt_cpp_api SHARED
        src/engine.cpp)

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})#${TensorRT_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR}
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES}) #${TensorRT_LIBRARIES}

add_executable(driver src/main.cpp)
target_link_libraries(driver tensorrt_cpp_api)