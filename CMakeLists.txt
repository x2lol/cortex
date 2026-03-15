cmake_minimum_required(VERSION 3.20)

project(neural_network_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(
    math
    nn
    datasets
    training
    serialization
    exceptions
)

set(SOURCES
    main.cpp

    datasets/loader.cpp

    nn/network.cpp
    nn/dense_layer.cpp

    training/trainer.cpp

    serialization/model_io.cpp
)

add_executable(network ${SOURCES})