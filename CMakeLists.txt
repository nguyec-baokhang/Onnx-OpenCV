cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(YoloOnnxC++)

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime-osx-arm64-1.23.2") 
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

find_package(OpenCV REQUIRED)

set(SOURCES ${CMAKE_SOURCE_DIR}/src/main.cpp 
        ${CMAKE_SOURCE_DIR}/src/munkres.cpp)

add_executable(object_detection ${SOURCES})
target_include_directories(object_detection PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(object_detection PUBLIC onnxruntime ${OpenCV_LIBS} )

set_property(TARGET object_detection PROPERTY CXX_STANDARD 20)
set_target_properties(object_detection PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)