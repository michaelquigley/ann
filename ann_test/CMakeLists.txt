if ("${CMAKE_SYSTEM_NAME}" MATCHES "Windows")
    set(CMAKE_EXE_LINKER_FLAGS "/SUBSYSTEM:CONSOLE")
endif ()

add_definitions(-DANN_NO_RANDOM)
add_executable(ann_test
    ann_test.cpp
    rand.cpp
    rand.h
)

target_link_libraries(ann_test
    ann
)