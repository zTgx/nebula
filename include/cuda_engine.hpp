#pragma once

#include <boost/json.hpp>

class CudaEngine {
public:
    CudaEngine();
    ~CudaEngine();
    boost::json::value compute(const boost::json::value& request);

private:
    void matrix_multiply(const float* a, const float* b, float* c, int m, int n, int k);
};