#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_engine.hpp"


// void CudaEngine::matrix_multiply(const float* a, const float* b, float* c, int m, int n, int k) {
//     const float alpha = 1.0f, beta = 0.0f;
//     cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta, c, n);
// }

// #include <iostream>
// #include <cublas_v2.h> 
// #include "cuda_error_check.h"
// #include <thrust/device_vector.h>
// #include "handle.hpp"
// #include "cuda_engine.hpp"

// __device__ int multiply(int x, int y) {
//    return x * y;
// }

// __global__ void addx(int *a, int *b, int *c) {
//     *c = *a + *b;
 
//     // This function acts as a barrier at which all threads in a block must wait until every thread reaches that point in the code. 
//     __syncthreads();
 
//     // __syncthreads(); // Ensure all additions are complete before writing back 
 
// }

// The __global__ qualifier indicates that this function runs on the device (GPU) and can be called from the host (CPU).
// __global__ declaration specifier, marking it as a function that runs on the GPU but can be called from the host and executed in parallel.

//1. must return void
//2. must have a unique name
//3. must be declared with the __global__ qualifier
//4. must be called from the host code using the <<<...>>> syntax, which specifies the number of blocks and threads per block.
//5. can take arguments, but they must be pointers to device memory or built-in types (like int, float, etc.).
//6. can use built-in variables like threadIdx, blockIdx, blockDim, and gridDim to determine the thread and block indices.
//7. can use synchronization functions like __syncthreads() to coordinate threads within a block.
// __global__ void kernel() {

//     // threadIdx.x is a built-in variable in CUDA that provides the thread ID within a block. 
//     // blockIdx.x is a built-in variable that provides the block ID within the grid.
//     printf("Hello from GPU thread %d in block %d\n", 
//            threadIdx.x, blockIdx.x);

   
//    // calc thread id 
//    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//    printf("Thread ID: %d\n", thread_id);

//    // call device function
//    int x = 2, y = 3;
//    int result = multiply(x, y);
//    printf("Multiplication result: %d\n", result);
// }

// #define N 512 
 
// __global__ void add(int *a, int *b, int *c) { 
//    int index = threadIdx.x; 
//    c[index] = a[index] + b[index]; 
// } 

