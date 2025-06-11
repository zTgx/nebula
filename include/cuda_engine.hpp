#pragma once

#include <boost/json.hpp>
#include <cublas_v2.h>

class CudaEngine {
public:
    CudaEngine();
    ~CudaEngine();
    boost::json::value compute(const boost::json::value& request);

private:
    void matrix_multiply(const float* a, const float* b, float* c, int m, int n, int k);
    void matrix_vector_multiply(const float* a, const float* x, float* y, int m, int n);
    void vector_dot(const float* x, const float* y, float* result, int n);

private:
    cublasHandle_t cublas_handle_ = nullptr;
};