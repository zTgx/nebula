cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME "nebula")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# MX150的计算能力是 ​​6.1（Maxwell架构）​​
# 从CUDA 12.x开始，NVIDIA逐步淘汰对 ​​计算能力 < 7.5（Volta+）​​ 的离线编译支持
# nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

set(CMAKE_CUDA_ARCHITECTURES "61")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# 查找CUDA库
find_package(CUDAToolkit REQUIRED)

# 使用GLOB查找所有的.cu文件
file(GLOB CUDA_SOURCES "src/*.cu")

# 添加可执行文件，并更改名称为${PROJECT_NAME}
add_executable(${PROJECT_NAME} ${CUDA_SOURCES})

# 链接cuBLAS和其他必要库
target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart      # CUDA运行时
    CUDA::cublas      # cuBLAS库（新增）
    CUDA::cublasLt    # cuBLAS扩展（可选）
)

# 针对MX150的优化选项
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall
    --default-stream per-thread
    >
)