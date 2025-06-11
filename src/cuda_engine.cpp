
#include <iostream>
#include "handle.hpp"
#include "cuda_engine.hpp"
#include <cuda_runtime.h>
#include <boost/json.hpp>

// #include <cuda_runtime.h>
// #include <cublas_v2.h>

CudaEngine::CudaEngine() {
   //  cublasCreate(&cublas_handle_);
}

CudaEngine::~CudaEngine() {
   //  cublasDestroy(cublas_handle_);
}

boost::json::value CudaEngine::compute(const boost::json::value& request) {
    std::cout << "Received request: " << boost::json::serialize(request) << std::endl;
    
    // auto task = request.at("task").as_string();
    // if (task == "matrix_multiply") {
    //     auto data = request.at("data");
    //     auto matrix_a = data.at("matrix_a").as_array();
    //     auto matrix_b = data.at("matrix_b").as_array();
    //     int m = matrix_a.size(), n = matrix_b[0].as_array().size(), k = matrix_b.size();

    //     std::vector<float> a(m * k), b(k * n), c(m * n);
    //     // 填充矩阵数据...

    //     float *d_a, *d_b, *d_c;
    //     cudaMalloc(&d_a, m * k * sizeof(float));
    //     cudaMalloc(&d_b, k * n * sizeof(float));
    //     cudaMalloc(&d_c, m * n * sizeof(float));

    //     cudaMemcpy(d_a, a.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_b, b.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);

    //     matrix_multiply(d_a, d_b, d_c, m, n, k);

    //     cudaMemcpy(c.data(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    //     cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    //     boost::json::array result;
    //     // 填充结果到 JSON...

    //     return boost::json::value{{"result", result}};
    // }

    return boost::json::value{{"error", "unknown task"}};
}