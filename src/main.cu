#include <iostream>
#include <cublas_v2.h> 
#include "cuda_error_check.h"
#include <thrust/device_vector.h>

__device__ int multiply(int x, int y) {
   return x * y;
}

// The __global__ qualifier indicates that this function runs on the device (GPU) and can be called from the host (CPU).
// __global__ declaration specifier, marking it as a function that runs on the GPU but can be called from the host and executed in parallel.

//1. must return void
//2. must have a unique name
//3. must be declared with the __global__ qualifier
//4. must be called from the host code using the <<<...>>> syntax, which specifies the number of blocks and threads per block.
//5. can take arguments, but they must be pointers to device memory or built-in types (like int, float, etc.).
//6. can use built-in variables like threadIdx, blockIdx, blockDim, and gridDim to determine the thread and block indices.
//7. can use synchronization functions like __syncthreads() to coordinate threads within a block.
__global__ void kernel() {

    // threadIdx.x is a built-in variable in CUDA that provides the thread ID within a block. 
    // blockIdx.x is a built-in variable that provides the block ID within the grid.
    printf("Hello from GPU thread %d in block %d\n", 
           threadIdx.x, blockIdx.x);

   
   // calc thread id 
   int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
   printf("Thread ID: %d\n", thread_id);

   // call device function
   int x = 2, y = 3;
   int result = multiply(x, y);
   printf("Multiplication result: %d\n", result);
}

#define N 512 
 
__global__ void add(int *a, int *b, int *c) { 
   int index = threadIdx.x; 
   c[index] = a[index] + b[index]; 
} 

void cublasExample() { 
   cublasHandle_t handle; 
   cublasCreate(&handle); 
 
   float alpha = 1.0f; 
   float beta = 0.0f; 
   int NN = 1024;
 
   // using d_ as a prefix to indicate that these variables are device pointers.
   float* d_A; 
   float* d_B; 
   float* d_C; 
   cudaMalloc((void**)&d_A, NN * NN * sizeof(float)); 
   cudaMalloc((void**)&d_B, NN * NN * sizeof(float)); 
   cudaMalloc((void**)&d_C, NN * NN * sizeof(float)); 
 
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, NN); 
 
   cublasDestroy(handle); 
   cudaFree(d_A); 
   cudaFree(d_B); 
   cudaFree(d_C); 
}

__global__ void addx(int *a, int *b, int *c) {
   *c = *a + *b;

   // This function acts as a barrier at which all threads in a block must wait until every thread reaches that point in the code. 
   __syncthreads();

   // __syncthreads(); // Ensure all additions are complete before writing back 

}

int main() {
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "Launching on: " << prop.name << "\n";

    thrust::device_vector<int> data(N);
    
    // Launch 4 parallel thread blocks
    // The first 2 indicates that the kernel will be launched with 2 blocks.
    // The second 2 indicates that each block will contain 2 threads.
    // <numBlocks, threadsPerBlock>
    kernel<<<2, 2>>>();

    // ensures the CPU waits for the GPU to complete execution before terminating the program.
    cudaDeviceSynchronize();

    std::cout << "Done!\n";

    int a[N], b[N], c[N]; 
    int *d_a, *d_b, *d_c; 
  
    // allocates memory on the GPU.
    cudaError_t mallocErr = cudaMalloc((void **) &d_a, N * sizeof(int)); 
    checkCudaErrors(mallocErr);

    cudaMalloc((void **) &d_b, N * sizeof(int)); 
    cudaMalloc((void **) &d_c, N * sizeof(int)); 
  
    for (int i = 0; i < N; i++) { 
       a[i] = i; 
       b[i] = i * i; 
    } 
  
    // copies data from the host (CPU) to the device (GPU).
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice); 
  
    // Launches the kernel with 1 block and N threads.
    add<<<1, N>>>(d_a, d_b, d_c); 
  
    // copies the result from the device (GPU) back to the host (CPU).
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost); 
  
    for (int i = 0; i < N; i++) { 
       printf("%d + %d = %d\n", a[i], b[i], c[i]); 
    } 
  
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); 

   //  std::cout << "CUDA example completed.\n";

   //  cublasExample();
   //  std::cout << "cuBLAS example completed.\n";
    
   cudaError_t err = cudaGetLastError(); 
   if (err != cudaSuccess) { 
      printf("CUDA Kernel launch failed: %s\n", cudaGetErrorString(err)); 
   }

   return 0;
}