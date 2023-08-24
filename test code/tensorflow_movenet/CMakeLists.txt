cmake_minimum_required(VERSION 3.16)
project(tensorflow_movenet)

set(CMAKE_CXX_STANDARD 17)

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

set(CMAKE_SYSTEM_PROCESSOR aarch64)

#include openCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} inc numpyC++)

add_executable(${PROJECT_NAME} src/main.cpp src/movenet_tools.cpp src/visualisation_tools.cpp src/classifier.cpp numpyC++/numpy_tools.cpp)

target_link_libraries(${PROJECT_NAME} tensorflow-lite ${OpenCV_LIBS})
