cmake_minimum_required(VERSION 3.2)
project(face_detection)

add_executable(
        face_detection
        src/face_recognition.h
        src/anchor_generator.cpp
        src/config.cpp
        demo.cpp
        src/tools.cpp
        src/anchor_generator.h
        src/config.h
        src/tools.h
        src/aligner.h
        src/aligner.cpp
)
set_target_properties(face_detection PROPERTIES CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenmp")
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
include_directories(
        ${PROJECT_SOURCE_DIR}/dependencies/ncnn/install/include/ncnn
        -I/usr/include/opencv4
)

target_link_libraries(
        face_detection
        ncnn
        -lstdc++fs
        ${OpenCV_LIBS}
)

add_library(ncnn STATIC IMPORTED)
set_property(TARGET ncnn PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/dependencies/ncnn/install/lib/libncnn.a)
