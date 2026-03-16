cmake_minimum_required(VERSION 3.20)

project(neural_network_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(
    src
)

set(SOURCES
    main.cpp

    src/cortex/datasets/loader.cpp

    src/cortex/nn/network.cpp
    src/cortex/nn/dense_layer.cpp

    src/cortex/training/trainer.cpp

    src/cortex/serialization/model_io.cpp
)

add_executable(network ${SOURCES})

target_include_directories(network PUBLIC src)