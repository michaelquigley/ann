if ("${CMAKE_SYSTEM_NAME}" MATCHES "Windows")
    set(CMAKE_EXE_LINKER_FLAGS "/SUBSYSTEM:CONSOLE")
endif ()

add_executable(ann_knn_01
    ann_knn_01.cpp
)

target_link_libraries(ann_knn_01
    ann
)