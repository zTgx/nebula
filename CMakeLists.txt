cmake_minimum_required(VERSION 3.18)
project(CudaNebula LANGUAGES CXX CUDA)

# 明确指定计算能力为6.1（MX150的架构）
set(CMAKE_CUDA_ARCHITECTURES "61")  # 关键修改！数字格式而非sm_xx

# 其他配置保持不变
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(cuda_nebula src/hello_gpu.cu)

# 优化编译选项
target_compile_options(cuda_nebula PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall
    --default-stream per-thread
    >
)