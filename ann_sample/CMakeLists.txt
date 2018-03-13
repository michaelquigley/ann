if ("${CMAKE_SYSTEM_NAME}" MATCHES "Windows")
    set(CMAKE_EXE_LINKER_FLAGS "/SUBSYSTEM:CONSOLE")
endif ()

add_executable(ann_sample
    ann_sample.cpp
)

target_link_libraries(ann_sample
    ann
)