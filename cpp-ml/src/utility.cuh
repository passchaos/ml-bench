#ifndef UTIL_H
#define UTIL_H

#include <algorithm>
#include <vector>
#include <iostream>
#include <random>

#define CHECK_CUDA(err) { \
    cudaError_t err_ = err; \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
}

namespace util {

void prepareRandomNumbersCpuGpu(unsigned int N,
    std::vector<float>& vals,
    float** dValsPtr) {
        constexpr float target = 2.0;

        std::cout << "Expected value: " << target * N << "\n";

        std::default_random_engine eng(0xcaffe);
        std::normal_distribution<float> dist(target);
        vals.resize(N);
        std::for_each(vals.begin(), vals.end(), [&dist, &eng](float& f) {
            f = dist(eng);
        });

        CHECK_CUDA(cudaMalloc((void **)dValsPtr, sizeof(float) * N));
        CHECK_CUDA(cudaMemcpy(*dValsPtr, vals.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    }

    __device__ void WasteTime(unsigned long long duration)
    {
        const unsigned long long start = clock64();
        while ((clock64() - start) < duration);
    }
}

#endif
