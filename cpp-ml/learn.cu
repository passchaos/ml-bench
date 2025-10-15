#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstring>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

void deviceInfo() {
    int numDevices;
    cudaGetDeviceCount(&numDevices);

    int device;
    cudaGetDevice(&device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    std::cout << "Number of devices: " << numDevices << std::endl;
    std::cout << "Current device: " << device << std::endl;
    std::cout << "Device name: " << props.name << std::endl;
    std::cout << "Device compute capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "Device total memory: " << props.totalGlobalMem / float(1<<30) << " GB" << std::endl;
    std::cout << "Device clock rate: " << props.clockRate / 1000000.0 << " GHz" << std::endl;
    std::cout << "Device multi-processor count: " << props.multiProcessorCount << std::endl;

}

__global__ void HelloWorldGpu() {
    printf("Hello world from c++ cuda\n");
}

void BasicExam() {
    deviceInfo();
    HelloWorldGpu<<<1, 12>>>();
    cudaError_t err = cudaDeviceSynchronize();
    printf("cuda synchronize res: %s\n", cudaGetErrorString(err));
}

__global__ void PrintIDs() {
    printf("Block ID: %d, %d, %d - Thread ID: %d, %d, %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

void GridExam() {
    const dim3 gridSize_small { 1, 1, 1};
    const dim3 blockSize_small{4, 3, 1};

    PrintIDs<<<gridSize_small, blockSize_small>>>();
    cudaDeviceSynchronize();
    printf("========================================\n");

    const dim3 gridSize_large { 2, 3, 1};
    const dim3 blockSize_large{4, 3, 1};

    PrintIDs<<<6, blockSize_large>>>();
    cudaDeviceSynchronize();
}

__device__ unsigned int step = 0;
__device__ void takeNTurns(const char*who, unsigned int N)
{
    int lastTurn = -42;
    int turn, start;

    for (int i = 0; i < N; i++) {
        int a = 0xFFFFFFFFU;
        printf("val: %d\n", a);
        turn = atomicAdd(&step, 0xFFFFFFFFU);
        printf("turn: %d\n", turn);

        bool switchOccurred = (lastTurn != (turn - 1));
        bool done = (i == (N - 1));

        if (done || switchOccurred) {
            printf("gotten %s: %d--%d\n", who, start, lastTurn + (done ? 1 : 0));
        }

        if (switchOccurred) {
            start = turn;
        }

        lastTurn = turn;
    }
}

__global__ void testScheduling(int N)
{
    if (threadIdx.x < 2) {
        if (threadIdx.x == 0) {
            takeNTurns("thread 0", N);
        } else {
            takeNTurns("thread 1", N);
        }
    } else {
        if (threadIdx.x == 2) {
            takeNTurns("thread 2", N);
        } else {
            takeNTurns("thread 3", N);
        }
    }
}

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

        cudaMalloc((void **)dValsPtr, sizeof(float) * N);
        cudaMemcpy(*dValsPtr, vals.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    }

__device__ float dResult;
__global__ void reduceAtomicGlobal(const float *__restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < N)
    {
        atomicAdd(&dResult, input[id]);
    }
}

__global__ void reduceAtomicShared(const float *__restrict input, int N)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float x;
    if (threadIdx.x == 0)
    {
        x = 0.0;
    }

    __syncthreads();

    if (id < N)
    {
        atomicAdd(&x, input[id]);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(&dResult, x);
    }
}

template <unsigned int BLOCK_SIZE>
__global__ void reduceShared(const float*__restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float data[BLOCK_SIZE];

    data[threadIdx.x] = (id < N ? input[id] : 0.0);

    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        __syncthreads();
        if (threadIdx.x < s)
        {
            data[threadIdx.x] += data[threadIdx.x + s];
        }
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&dResult, data[0]);
    }
}

void ReduceExam() {
    constexpr unsigned int BLOCK_SIZE = 256;
    constexpr unsigned int WARMUP_ITERATIONS = 10;

    constexpr unsigned int TIMING_ITERATIONS = 10;

    constexpr unsigned int N = 10'000'000;

    std::cout << "Producing random inputs...\n" << std::endl;

    std::vector<float> vals;
    float* dValsPtr;
    prepareRandomNumbersCpuGpu(N, vals, &dValsPtr);

    const auto before = std::chrono::system_clock::now();
    // auto before_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(before).time_since_epoch().count();

    auto cpu_val = std::accumulate(vals.begin(), vals.end(), 0.0);
    const auto after = std::chrono::system_clock::now();
    // auto after_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(after).time_since_epoch().count();

    std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count() << " ns\n";

    const std::tuple<const char*, void(*)(const float*, int), unsigned int> reductionTqs[] {
        {"Atomic Global", reduceAtomicGlobal, N},
        {"Atomic Shared", reduceAtomicShared, N},
        {"Reduce Shared", reduceShared<BLOCK_SIZE>, N},
    };

    for (const auto& [name, func, numThreads] : reductionTqs)
    {
        const auto before = std::chrono::system_clock::now();

        float result = 0.0;
        cudaMemcpyToSymbol(dResult, &result, sizeof(float));

        const dim3 blockDim = BLOCK_SIZE;
        const dim3 gridDim = (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        func<<<gridDim, blockDim>>>(dValsPtr, N);
        cudaMemcpyFromSymbol(&result, dResult, sizeof(float));
        cudaDeviceSynchronize();

        float diff = cpu_val - result;

        const auto after = std::chrono::system_clock::now();

        std::cout << name << " time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count() << " ns " << "val diff: " << result << "\n";
    }

    cudaFree(dValsPtr);
}


int main() {
    // GridExam();
    // takeNTurns<<<1, 1>>>("main", 5);
    // testScheduling<<<1, 4>>>(5);
    ReduceExam();

    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    return 0;
}
