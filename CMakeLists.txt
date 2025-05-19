cmake_minimum_required(VERSION 3.18)
project(nebula LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "61")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(nebula src/hello_gpu.cu)

target_compile_options(nebula PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall
    --default-stream per-thread
    >
)