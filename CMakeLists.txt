cmake_minimum_required(VERSION 2.6)

project(nanodet)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda/include)
include_directories(/home/yc-mc/Downloads/TensorRT-8.2.0.6/include)
link_directories(/home/yc-mc/Downloads/TensorRT-8.2.0.6/lib)
link_directories(/usr/local/cuda/lib64)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(nanodet ${PROJECT_SOURCE_DIR}/nanodet.cpp)
target_link_libraries(nanodet nvinfer)
target_link_libraries(nanodet cudart)
target_link_libraries(nanodet ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

