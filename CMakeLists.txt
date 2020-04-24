cmake_minimum_required(VERSION 3.10)

project(ann)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(ANN)
add_subdirectory(ann_fig)
add_subdirectory(ann_knn_01)
add_subdirectory(ann_sample)
add_subdirectory(ann_test)