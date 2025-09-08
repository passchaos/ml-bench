#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <chrono>

// 错误检查宏定义
#define CHECK_CUDA(status) \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUBLAS(status) \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CURAND(status) \
    if (status != CURAND_STATUS_SUCCESS) { \
        std::cerr << "cuRAND error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// 生成均匀分布的随机矩阵
void generate_uniform_matrix(float* d_matrix, int rows, int cols, curandGenerator_t& gen, float min_val = 0.0f, float max_val = 1.0f) {
    // 生成[0,1)范围的随机数
    CHECK_CURAND(curandGenerateUniform(gen, d_matrix, rows * cols));

    // 转换到[min_val, max_val)范围
    float range = max_val - min_val;
    float* h_scale = new float[1];
    float* h_bias = new float[1];
    h_scale[0] = range;
    h_bias[0] = min_val;

    float* d_scale;
    float* d_bias;
    CHECK_CUDA(cudaMalloc(&d_scale, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_scale, h_scale, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias, sizeof(float), cudaMemcpyHostToDevice));

    // 使用cuBLAS执行缩放: d_matrix = d_matrix * scale + bias
    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUBLAS(cublasSscal(cublas, rows * cols, d_scale, d_matrix, 1));
    CHECK_CUBLAS(cublasSaxpy(cublas, rows * cols, d_bias, nullptr, 0, d_matrix, 1));
    CHECK_CUBLAS(cublasDestroy(cublas));

    // 清理临时变量
    delete[] h_scale;
    delete[] h_bias;
    CHECK_CUDA(cudaFree(d_scale));
    CHECK_CUDA(cudaFree(d_bias));
}

int main() {
    // 矩阵维度定义
    const int M = 60000;  // A的行数，结果的行数
    const int K = 784;    // A的列数，B的行数
    const int N = 1000;   // B的列数，C的列数

    // 初始化cuBLAS和cuRAND
    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUBLAS(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));  // 启用Tensor Core加速

    std::cout << "cuBLAS initialized with Tensor Core acceleration." << std::endl;

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)std::chrono::system_clock::now().time_since_epoch().count()));

    // 分配设备内存
    float *d_A, *d_B, *d_C, *d_D;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));  // 60000x784
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));  // 784x1000
    CHECK_CUDA(cudaMalloc(&d_C, 1 * N * sizeof(float)));   // 1x1000
    CHECK_CUDA(cudaMalloc(&d_D, M * N * sizeof(float)));  // 结果矩阵 60000x1000

    // 生成均匀分布的随机矩阵
    std::cout << "Generating random matrices..." << std::endl;
    generate_uniform_matrix(d_A, M, K, gen);
    generate_uniform_matrix(d_B, K, N, gen);
    generate_uniform_matrix(d_C, 1, N, gen);

    // 预热运行 - 确保所有内核加载完成
    std::cout << "Warming up..." << std::endl;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 矩阵乘法: D = A * B
    CHECK_CUBLAS(cublasSgemm(cublas,
                           CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置A和B
                           N, M, K,                  // 计算维度 (N, M, K)
                           &alpha,
                           d_B, N,                   // B矩阵及领先维度
                           d_A, K,                   // A矩阵及领先维度
                           &beta,
                           d_D, N));                 // 结果矩阵D及领先维度

    // 矩阵加法: D = D + C (利用广播机制)
    // 由于C是1xN，我们需要将其广播到M行，这里通过重复加法实现
    for (int i = 0; i < M; ++i) {
        CHECK_CUBLAS(cublasSaxpy(cublas, N, &alpha, d_C, 1, d_D + i * N, 1));
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // 性能测试
    std::cout << "Benchmarking..." << std::endl;
    const int num_runs = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < num_runs; ++run) {
        // 矩阵乘法: D = A * B
        CHECK_CUBLAS(cublasSgemm(cublas,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K,
                               &alpha,
                               d_B, N,
                               d_A, K,
                               &beta,
                               d_D, N));

        // 矩阵加法: D = D + C (广播)
        for (int i = 0; i < M; ++i) {
            CHECK_CUBLAS(cublasSaxpy(cublas, N, &alpha, d_C, 1, d_D + i * N, 1));
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // 计算并输出平均时间
    std::chrono::duration<double, std::milli> avg_time = (end - start) / num_runs;
    std::cout << "Average time over " << num_runs << " runs: " << avg_time.count() << " ms" << std::endl;

    // 资源清理
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUBLAS(cublasDestroy(cublas));
    CHECK_CURAND(curandDestroyGenerator(gen));

    return 0;
}
