#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 定义颜色代码用于终端输出
#define RED   "\x1B[31m"
#define GREEN "\x1B[32m"
#define YELLOW "\x1B[33m"
#define BLUE  "\x1B[34m"
#define MAGENTA "\x1B[35m"
#define CYAN  "\x1B[36m"
#define WHITE "\x1B[37m"
#define RESET "\x1B[0m"

// 获取CUDA错误字符串的辅助函数
static const char* _cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

// 详细的CUDA错误检查宏
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// 内联函数处理错误
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, RED "CUDA error: %s %s at %s:%d\n" RESET, 
                _cudaGetErrorEnum(err), cudaGetErrorString(err), file, line);
        
        // 打印更详细的调试信息
        fprintf(stderr, YELLOW "Debug info:\n");
        fprintf(stderr, "  Error code: %d\n", err);
        fprintf(stderr, "  Error name: %s\n", _cudaGetErrorEnum(err));
        fprintf(stderr, "  Error description: %s\n" RESET, cudaGetErrorString(err));
        
        // 建议的解决方案
        fprintf(stderr, CYAN "Possible solutions:\n");
        switch(err) {
            case cudaErrorMemoryAllocation:
                fprintf(stderr, "  - Check if you have enough GPU memory\n");
                fprintf(stderr, "  - Try reducing the memory requirements\n");
                break;
            case cudaErrorInvalidValue:
                fprintf(stderr, "  - Check for invalid parameters passed to CUDA functions\n");
                break;
            case cudaErrorInvalidDevicePointer:
                fprintf(stderr, "  - Check if device pointers are properly initialized\n");
                break;
            case cudaErrorInvalidConfiguration:
                fprintf(stderr, "  - Check kernel launch configuration (block/grid size)\n");
                break;
            default:
                fprintf(stderr, "  - Refer to CUDA documentation for error code %d\n", err);
        }
        fprintf(stderr, RESET);
        
        // 确保错误信息被刷新
        fflush(stderr);
        
        // 退出程序或抛出异常
        exit(EXIT_FAILURE);
    }
}

// 同步设备并检查错误的宏
#define checkCudaKernelError() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, RED "Kernel launch error: %s at %s:%d\n" RESET, \
               cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
    err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, RED "Kernel execution error: %s at %s:%d\n" RESET, \
               cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#endif // CUDA_ERROR_CHECK_H