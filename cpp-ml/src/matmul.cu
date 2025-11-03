#include <cstdio>
#include <iostream>
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
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int m, int n, int k, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  const uint x = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
  const uint y = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

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

int main() {
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
  // sgemm_naive<<<gridDim, blockDim>>>(m, n, k, 2.0, a, b, 0.0, c1);
  // sgemm_global_mem_coalesce<32>
  //     <<<gridDim, 32 * 32>>>(m, n, k, 2.0, a, b, 0.0, c2);

  // sgemm_naive_transpose<<<gridDim, blockDim>>>(m, n, k, 2.0, a, b, 0.0, c2);
  sgemm_naive_intuitive<<<gridDim, blockDim>>>(m, n, k, 2.0, a, b, 0.0, c2);
  cudaEventRecord(compute_end);

  // cudaMemcpy(v_c1.data(), c1, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaMemcpy(v_c2.data(), c2, m * n * sizeof(float), cudaMemcpyDeviceToHost);
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
