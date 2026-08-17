cmake_minimum_required(VERSION 3.24)
project(cudanatron VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "75")

option(CUDANATRON_BUILD_TESTS "Build cudanatron tests" ON)

add_library(cudanatron
    src/map.cu
    src/game.cu
    src/action_space.cu
    src/observation.cu
    src/chance.cu
    src/mcts.cpp
    src/batch.cu
)
add_library(cudanatron::cudanatron ALIAS cudanatron)

set_target_properties(cudanatron PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(
    cudanatron
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cudanatron PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cudanatron PUBLIC OpenMP::OpenMP_CXX)
endif()

add_library(cudanatron_c SHARED src/c_api.cu)
target_link_libraries(cudanatron_c PRIVATE cudanatron::cudanatron)
set_target_properties(cudanatron_c PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    OUTPUT_NAME cudanatron
)
target_include_directories(cudanatron_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CUDANATRON_BUILD_TESTS)
    enable_testing()
    add_executable(cudanatron_tests tests/test_engine.cpp)
    target_link_libraries(cudanatron_tests PRIVATE cudanatron::cudanatron)
    add_test(NAME cudanatron_tests COMMAND cudanatron_tests)
endif()
