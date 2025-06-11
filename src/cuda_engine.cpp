
#include <iostream>
#include "handle.hpp"
#include "cuda_engine.hpp"
#include <cuda_runtime.h>
#include <boost/json.hpp>
#include <cublas_v2.h>

CudaEngine::CudaEngine() {
   cublasStatus_t status = cublasCreate(&cublas_handle_);
   if (status != CUBLAS_STATUS_SUCCESS) {
       std::cerr << "cuBLAS init failed: " << cublasGetStatusString(status) << std::endl;
       throw std::runtime_error("cuBLAS initialization failed");
   }
   std::cout << "cuBLAS initialized successfully." << std::endl;
}

CudaEngine::~CudaEngine() {
   cublasDestroy(cublas_handle_);
}

boost::json::value CudaEngine::compute(const boost::json::value& request) {
   try {
       auto task = request.at("task").as_string();
       auto data = request.at("data");

       if (task == "matrix_multiply") {
           // Get matrix_a and matrix_b as arrays
           auto matrix_a = data.at("matrix_a").as_array();
           auto matrix_b = data.at("matrix_b").as_array();
           int m = matrix_a.size(); // Rows of A
           int k = matrix_a[0].as_array().size(); // Columns of A
           int n = matrix_b[0].as_array().size(); // Columns of B

           // Validate dimensions
           if (matrix_b.size() != k) {
               throw std::runtime_error("Matrix dimensions mismatch");
           }

           // Allocate host memory
           std::vector<float> a(m * k), b(k * n), c(m * n);

           // Fill matrix_a
           for (int i = 0; i < m; ++i) {
               auto row = matrix_a[i].as_array(); // Convert to array
               if (row.size() != k) {
                   throw std::runtime_error("Invalid matrix_a row size");
               }
               for (int j = 0; j < k; ++j) {
                   a[i * k + j] = static_cast<float>(row[j].as_double());
               }
           }

           // Fill matrix_b
           for (int i = 0; i < k; ++i) {
               auto row = matrix_b[i].as_array();
               if (row.size() != n) {
                   throw std::runtime_error("Invalid matrix_b row size");
               }
               for (int j = 0; j < n; ++j) {
                   b[i * n + j] = static_cast<float>(row[j].as_double());
               }
           }

           // Allocate device memory
           float *d_a, *d_b, *d_c;
           cudaMalloc(&d_a, m * k * sizeof(float));
           cudaMalloc(&d_b, k * n * sizeof(float));
           cudaMalloc(&d_c, m * n * sizeof(float));

           // Copy data to device
           cudaMemcpy(d_a, a.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
           cudaMemcpy(d_b, b.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);

           // Perform matrix multiplication
           matrix_multiply(d_a, d_b, d_c, m, n, k);

           // Copy result back to host
           cudaMemcpy(c.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

           // Free device memory
           cudaFree(d_a);
           cudaFree(d_b);
           cudaFree(d_c);

           // Create JSON result
           boost::json::array result;
           for (int i = 0; i < m; ++i) {
               boost::json::array row;
               for (int j = 0; j < n; ++j) {
                   row.push_back(c[i * n + j]);
               }
               result.push_back(row);
           }
           return boost::json::value{{"result", result}};
       }
       return boost::json::value{{"error", "unknown task"}};
   } catch (const std::exception& e) {
       return boost::json::value{{"error", e.what()}};
   }
}

void CudaEngine::matrix_multiply(const float* a, const float* b, float* c, int m, int n, int k) {
   const float alpha = 1.0f, beta = 0.0f;
   cublasStatus_t status = cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta, c, n);
   if (status != CUBLAS_STATUS_SUCCESS) {
       throw std::runtime_error("cublasSgemm failed: " + std::string(cublasGetStatusString(status)));
   }
}

void CudaEngine::matrix_vector_multiply(const float* a, const float* x, float* y, int m, int n) {
   const float alpha = 1.0f, beta = 0.0f;
   cublasStatus_t status = cublasSgemv(cublas_handle_, CUBLAS_OP_N, n, m, &alpha, a, n, x, 1, &beta, y, 1);
   if (status != CUBLAS_STATUS_SUCCESS) {
       throw std::runtime_error("cublasSgemv failed: " + std::string(cublasGetStatusString(status)));
   }
}

void CudaEngine::vector_dot(const float* x, const float* y, float* result, int n) {
   cublasStatus_t status = cublasSdot(cublas_handle_, n, x, 1, y, 1, result);
   if (status != CUBLAS_STATUS_SUCCESS) {
       throw std::runtime_error("cublasSdot failed: " + std::string(cublasGetStatusString(status)));
   }
}