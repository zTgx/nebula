#include <iostream>

__global__ void kernel() {
    printf("Hello from GPU thread %d in block %d\n", 
           threadIdx.x, blockIdx.x);
}

int main() {
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "Launching on: " << prop.name << "\n";
    
    // Launch 4 parallel thread blocks
    kernel<<<2, 2>>>();
    cudaDeviceSynchronize();
    
    return 0;
}