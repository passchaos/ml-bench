#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <driver_types.h>
#include <iostream>
#include <library_types.h>
#include <vector>
// #include <cuda/std/chrono>

#include "utility.cuh"

// m=n=k=4096
// kernel cost: 56.7ms
// GFLOPs: 2416.2
__global__ void sgemm_naive_intuitive(int m, int n, int k, float alpha,
                                      const float *A, const float *B,
                                      float beta, float *C) {

  const uint idx = blockIdx.y * (gridDim.x * blockDim.y * blockDim.x) +
                   blockIdx.x * (blockDim.y * blockDim.x) +
                   threadIdx.y * blockDim.x + threadIdx.x;

  if (idx < m * n) {
    const uint x = idx / n;
    const uint y = idx % n;

    float tmp = 0.0;

    for (int i = 0; i < k; ++i) {
      tmp += A[x * k + i] * B[i * n + y];
    }

    C[x * n + y] = alpha * tmp + beta * C[x * n + y];
  }
}

// kernel cost: 32.7ms
__global__ void sgemm_naive_transpose(int m, int n, int k, float alpha,
                                      const float *A, const float *B,
                                      float beta, float *C) {
  const uint x = blockIdx.y * blockDim.y + threadIdx.y;
  const uint y = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < m && y < n) {
    float tmp = 0.0;

    for (int i = 0; i < k; ++i) {
      tmp += A[x * k + i] * B[i * n + y];
    }

    C[x * n + y] = alpha * tmp + beta * C[x * n + y];
  }
}

// kernel cost: 38.7ms
// 这里虽然BLOCKSIZE为一个模板常数参数，但值只能是32
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int m, int n, int k, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  const uint x = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
  const uint y = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  // max_x = gridDim.x * BLOCKSIZE + blockDim.x / BLOCKSIZE;
  // max_y = gridDim.y * BLOCKSIZE + blockDim.y / BLOCKSIZE;
  // max_x * max_y = gridDim.x * gridDim.y * BLOCKSIZE * BLOCKSIZE
  // + gridDim.y * blockDim.x + gridDim.x * blockDim.y
  // + blockDim.x * blockDim.y / BLOCKSIZE / BLOCKSIZE

  if (x < m && y < n) {
    float tmp = 0.0;

    for (int i = 0; i < k; ++i) {
      tmp += A[x * k + i] * B[i * n + y];
    }

    C[x * n + y] = alpha * tmp + beta * C[x * n + y];
  }
}

// m=n=k=4096
// kernel cost: 249.7ms
// GFLOPs: 548.8
__global__ void sgemm_naive(int m, int n, int k, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < m && y < n) {
    float tmp = 0.0;

    for (int i = 0; i < k; ++i) {
      tmp += A[x * k + i] * B[i * n + y];
    }

    C[x * n + y] = alpha * tmp + beta * C[x * n + y];
  }
}

int get_array_diff(std::vector<float> &v_a, std::vector<float> &v_b) {
  int diff = 0;
  for (int i = 0; i < v_a.size(); ++i) {
    diff += abs(v_a[i] - v_b[i]);
  }
  return diff;
}

typedef __nv_bfloat16 dft;

constexpr cudaDataType cbft = CUDA_R_16BF;

// f32=============================================================================
// only compute
// CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION: 2.91ms
// CUBLAS_PEDANTIC_MATH: 2.89ms
// CUBLAS_TENSOR_OP_MATH: 1.82ms
// CUBLAS_TENSOR_OP_MATH + result_copy: 1.93ms
//
//
// bf16===========================================================================
// CUBLAS_COMPUTE_32F_FAST_16BF or CUBLAS_COMPUTE_32F
// CUBlAS_GEMM_DEFAULT or CUBLAS_GEMM_DEFAULT_TENSOR_OP
// compute only: 0.91ms
// compute only + result_copy: 1.09ms
__global__ void fp32_to_bf16(const float *in, dft *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = __float2bfloat16(in[idx]);
  }
};

void generate_cuda_dft_array(int m, int k, int n, curandGenerator_t generator,
                             dft **a, dft **b, dft **c) {

  float *h_a, *h_b, *h_c;

  CHECK_CUDA(cudaMalloc(a, m * k * sizeof(dft)));
  CHECK_CUDA(cudaMalloc(b, k * n * sizeof(dft)));
  CHECK_CUDA(cudaMalloc(c, m * n * sizeof(dft)));

  CHECK_CUDA(cudaMalloc(&h_a, m * k * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&h_b, k * n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&h_c, m * n * sizeof(float)));

  // CHECK_CURAND(curandGenerateUniform(generator, *a, m * k));
  // CHECK_CURAND(curandGenerateUniform(generator, *b, k * n));
  // CHECK_CURAND(curandGenerateUniform(generator, *c, m * n));
  CHECK_CURAND(curandGenerateUniform(generator, h_a, m * k));
  CHECK_CURAND(curandGenerateUniform(generator, h_b, k * n));
  CHECK_CURAND(curandGenerateUniform(generator, h_c, m * n));

  int block_size = 256;
  int grid_size_A = (m * k + block_size - 1) / block_size;
  int grid_size_B = (k * n + block_size - 1) / block_size;
  int grid_size_C = (m * n + block_size - 1) / block_size;
  fp32_to_bf16<<<grid_size_A, block_size>>>(h_a, *a, m * k);
  fp32_to_bf16<<<grid_size_B, block_size>>>(h_b, *b, k * n);
  fp32_to_bf16<<<grid_size_C, block_size>>>(h_c, *c, m * n);

  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaFree(h_a));
  CHECK_CUDA(cudaFree(h_b));
  CHECK_CUDA(cudaFree(h_c));
}

void cublasDemo() {
  int m = 1024 * 4;
  int k = 1024 * 4;
  int n = 1024 * 4;

  dft *a;
  dft *b;
  dft *c;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  curandGenerator_t generator;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);

  // const dft alpha = 1.0;
  // const dft beta = 1.0;
  const dft alpha = __float2bfloat16(1.0);
  const dft beta = __float2bfloat16(1.0);

  generate_cuda_dft_array(m, k, n, generator, &a, &b, &c);
  // warmup
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                            a, cbft, m, b, cbft, k, &beta, c, cbft, m,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> costs;
  for (int i = 0; i < 1000; i++) {
    cudaEvent_t start, compute_end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&compute_end));

    // CHECK_CUDA(cudaMalloc(&a, m * k * sizeof(dft)));
    // CHECK_CUDA(cudaMalloc(&b, k * n * sizeof(dft)));
    // CHECK_CUDA(cudaMalloc(&c, m * n * sizeof(dft)));

    // // CHECK_CUDA(cudaMalloc(&h_a, m * k * sizeof(float)));
    // // CHECK_CUDA(cudaMalloc(&h_b, k * n * sizeof(float)));
    // // CHECK_CUDA(cudaMalloc(&h_c, m * n * sizeof(float)));

    // CHECK_CURAND(curandGenerateUniform(generator, a, m * k));
    // CHECK_CURAND(curandGenerateUniform(generator, b, k * n));
    // CHECK_CURAND(curandGenerateUniform(generator, c, m * n));

    generate_cuda_dft_array(m, k, n, generator, &a, &b, &c);

    auto begin = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                              a, cbft, m, b, cbft, k, &beta, c, cbft, m,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    // dft *c2;

    // CHECK_CUDA(cudaMalloc(&c2, m * n * sizeof(dft)));
    // CHECK_CUDA(
    //     cudaMemcpy(c2, c, m * n * sizeof(dft), cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaEventRecord(compute_end));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    float time_compute;

    cudaEventElapsedTime(&time_compute, start, compute_end);
    printf("Time for compute: %fms %ldms\n", time_compute, duration);
    costs.push_back(time_compute);

    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
    // CHECK_CUDA(cudaFree(c2));
  }

  auto max_it = std::max_element(costs.begin(), costs.end());
  costs.erase(max_it);

  auto min_it = std::min_element(costs.begin(), costs.end());
  costs.erase(min_it);

  auto mean_cost =
      std::accumulate(costs.begin(), costs.end(), 0.0) / costs.size();
  printf("cublas compute cost: %fms\n", mean_cost);

  cublasDestroy(handle);
  curandDestroyGenerator(generator);
}

int main() {
  cublasDemo();
  return 0;

  constexpr int m = 1024 * 4;
  constexpr int n = 1024 * 4;
  constexpr int k = 1024 * 4;
  std::vector<float> v_a(m * k, 1.0);
  std::vector<float> v_b(k * n, 1.0);
  std::vector<float> v_c1(m * n, 1.0);
  std::vector<float> v_c2(m * n, 1.0);

  dim3 gridDim((m + 32 - 1) / 32, (n + 32 - 1) / 32, 1);
  dim3 blockDim(32, 32, 1);

  float *a;
  float *b;
  float *c1;
  float *c2;

  util::prepareRandomNumbersCpuGpu(m * k * sizeof(float), v_a, &a);
  util::prepareRandomNumbersCpuGpu(k * n * sizeof(float), v_b, &b);
  util::prepareRandomNumbersCpuGpu(m * n * sizeof(float), v_c1, &c1);
  util::prepareRandomNumbersCpuGpu(m * n * sizeof(float), v_c2, &c2);

  cudaEvent_t start, compute_end, copy_end;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&compute_end));
  CHECK_CUDA(cudaEventCreate(&copy_end));

  cudaEventRecord(start);
  sgemm_naive<<<gridDim, blockDim>>>(m, n, k, 1.0, a, b, 1.0, c1);

  // sgemm_global_mem_coalesce<32>
  //     <<<gridDim, 32 * 32>>>(m, n, k, 1.0, a, b, 1.0, c2);

  // sgemm_naive_transpose<<<gridDim, blockDim>>>(m, n, k, 2.0, a, b, 0.0,
  // c2); sgemm_naive_intuitive<<<gridDim, blockDim>>>(m, n, k, 2.0, a, b,
  // 0.0, c2);
  cudaEventRecord(compute_end);

  cudaMemcpy(v_c1.data(), c1, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(v_c2.data(), c2, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventRecord(copy_end);

  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  float time_compute, time_copy;
  cudaEventElapsedTime(&time_compute, start, compute_end);
  cudaEventElapsedTime(&time_copy, compute_end, copy_end);

  std::cout << "Time for computation: " << time_compute << " ms" << std::endl;
  std::cout << "Time for copy: " << time_copy << " ms" << std::endl;

  auto diff = get_array_diff(v_c1, v_c2);
  std::cout << "Difference between results: " << diff << std::endl;

  return 0;
}
