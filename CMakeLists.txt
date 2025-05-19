cmake_minimum_required(VERSION 3.18)
project(nebula LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "61")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# 查找CUDA库
find_package(CUDAToolkit REQUIRED)

# 添加可执行文件
add_executable(nebula src/hello_gpu.cu)

# 链接cuBLAS和其他必要库
target_link_libraries(nebula PRIVATE
    CUDA::cudart      # CUDA运行时
    CUDA::cublas      # cuBLAS库（新增）
    CUDA::cublasLt    # cuBLAS扩展（可选）
)

# 针对MX150的优化选项
target_compile_options(nebula PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall
    --default-stream per-thread
    >
)