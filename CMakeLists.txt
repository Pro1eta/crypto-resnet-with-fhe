cmake_minimum_required(VERSION 3.10)
project(FHEResNet)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)

# OpenFHE 构建目录可通过 -DOPENFHE_BUILD_DIR=... 传入
if(DEFINED OPENFHE_BUILD_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${OPENFHE_BUILD_DIR}")
endif()

find_package(OpenFHE CONFIG REQUIRED)
if(NOT OpenFHE_FOUND)
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif()

set(CMAKE_CXX_FLAGS "${OpenFHE_CXX_FLAGS} -Wno-uninitialized -Wno-maybe-uninitialized")

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(src)

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES})
else()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif()

set(SOURCES
    src/main.cpp
    src/fhe_context.cpp
    src/packing.cpp
    src/conv.cpp
    src/downsample.cpp
    src/avgpool.cpp
    src/activation.cpp
    src/fc.cpp
    src/resnet.cpp
    src/weight_loader.cpp
)

add_executable(FHEResNet ${SOURCES})
