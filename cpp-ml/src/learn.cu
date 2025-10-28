#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstring>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include "utility.cuh"

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
    std::cout << "Concurrent managed access: " << props.concurrentManagedAccess << std::endl;
    std::cout << "Device total memory: " << props.totalGlobalMem / float(1<<30) << " GB" << std::endl;
    // std::cout << "Device clock rate: " << props.clockRate / 1000000.0 << " GHz" << std::endl;
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

template <unsigned int BLOCK_SIZE>
__global__ void reduceShuffle(const float*__restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float data[BLOCK_SIZE];

    data[threadIdx.x] = (id < N ? input[id] : 0.0);

    for (int s = blockDim.x / 2; s > 16; s /= 2)
    {
        __syncthreads();
        if (threadIdx.x < s)
        {
            data[threadIdx.x] += data[threadIdx.x + s];
        }
    }

    float x = data[threadIdx.x];
    if (threadIdx.x < 32)
    {
        x += __shfl_down_sync(0xffffffff, x, 16);
        x += __shfl_down_sync(0xffffffff, x, 8);
        x += __shfl_down_sync(0xffffffff, x, 4);
        x += __shfl_down_sync(0xffffffff, x, 2);
        x += __shfl_down_sync(0xffffffff, x, 1);
    }


    if (threadIdx.x == 0)
    {
        atomicAdd(&dResult, x);
    }
}


template <unsigned int BLOCK_SIZE>
__global__ void reduceFinal(const float*__restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float data[BLOCK_SIZE];

    data[threadIdx.x] = (id < N ? input[id] : 0.0);
    data[threadIdx.x] += id + N/2 < N ? input[id + N/2] : 0.0;

    for (int s = blockDim.x / 2; s > 16; s /= 2)
    {
        __syncthreads();
        if (threadIdx.x < s)
        {
            data[threadIdx.x] += data[threadIdx.x + s];
        }
    }

    float x = data[threadIdx.x];
    if (threadIdx.x < 32)
    {
        x += __shfl_down_sync(0xffffffff, x, 16);
        x += __shfl_down_sync(0xffffffff, x, 8);
        x += __shfl_down_sync(0xffffffff, x, 4);
        x += __shfl_down_sync(0xffffffff, x, 2);
        x += __shfl_down_sync(0xffffffff, x, 1);
    }


    if (threadIdx.x == 0)
    {
        atomicAdd(&dResult, x);
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
        {"RecuceShuffle", reduceShuffle<BLOCK_SIZE>, N},
        {"RecuceFinal", reduceFinal<BLOCK_SIZE>, N / 2 + 1},
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

__global__ void SlowKernel() {
    util::WasteTime(1'000'000'000ULL);
    printf("I'm awake!\n");
}

void NsightExam() {
    cudaStream_t streams[5];
    for (int i = 0; i < 5; i++)
    {
        cudaStreamCreate(&streams[i]);
        SlowKernel<<<1, 1, 0, streams[i]>>>();

    }

    cudaDeviceSynchronize();
}

void EventExam() {
    cudaEvent_t start, end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaDeviceSynchronize();

    auto before = std::chrono::system_clock::now();
    cudaEventRecord(start);

    SlowKernel<<<1, 1>>>();

    cudaEventRecord(end);

    auto afterNoSync = std::chrono::system_clock::now();

    // cudaDeviceSynchronize();
    cudaEventSynchronize(end);

    auto afterSync = std::chrono::system_clock::now();

    std::cout << "No Sync Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(afterNoSync - before).count() << " ns\n";
    std::cout << "Sync Time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(afterSync - afterNoSync).count() << " ns\n";

    float msGPU;
    cudaEventElapsedTime(&msGPU, start, end);
    std::cout << "Measured time (CUDA events): " << msGPU << " ms\n";
    // cudaEventElapsedTime_v2(float *ms, cudaEvent_t start, cudaEvent_t end)
}

template<int offset>
__global__ void MemCoalesecd(int* in, int* out, size_t elements)
{
    size_t block_offset = blockIdx.x * blockDim.x;
    size_t warp_offset = 32 * (threadIdx.x / 32);
    size_t laneid = threadIdx.x % 32;
    // size_t laneid = (threadIdx.x * 7) % 32;

    size_t id = ((block_offset + warp_offset + laneid) * offset) % elements;

    out[id] = in[id];
}

void MemoryExam() {
    int* valsA;
    int* valsB;

    constexpr size_t size = 800'000'000'000;

    auto data_size = sizeof(int) * size;
    cudaMalloc((void **)&valsA, data_size);
    cudaMalloc((void **)&valsB, data_size);

    constexpr unsigned int BLOCK_SIZE = 256;
    const dim3 blockDim = BLOCK_SIZE;
    const dim3 gridDim = (size + blockDim.x - 1) / blockDim.x;

    auto before = std::chrono::system_clock::now();
    MemCoalesecd<35><<<blockDim, gridDim>>>(valsB, valsA, size);
    cudaDeviceSynchronize();
    auto after = std::chrono::system_clock::now();

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(after - before);
    auto bandwidth = data_size / (elapsed_ms.count() / 1000.0) / 1024.0 / 1024.0 / 1024.0;
    std::cout << "Elapsed time: " << elapsed_ms.count() << " ms " << bandwidth << " GB/s\n";
}

__global__ void SharedMemorySum() {
    __shared__ float mydata[32 * 32];

    float sum1 = 0;
    for (int i = 0; i < 32; i++) {
        sum1 += mydata[threadIdx.x * 32 + i];
    }
    printf("haha");

    auto start = clock64();
    float sum = 0;

    for (int i = 0; i < 32; i++) {
        sum += mydata[threadIdx.x * 32 + i];
    }

    auto step1 = clock64();

    sum = 0;

    for (int i = 0; i < 32; i++) {
        sum += mydata[threadIdx.x + i * 32];
    }

    auto step2 = clock64();

    printf("dur1: %lld du2: %lld\n", step1 - start, step2 - step1);
}

void SharedMemoryExam() {
    SharedMemorySum<<<1, 32>>>();
    cudaDeviceSynchronize();
}

// __managed__ int foo;
// __global__ void kernel(int *bar)
// {
//     // NOLINTNEXTLINE(clang-diagnostic-ref-bad-target)
//     printf("%d %X\n", foo, *bar);
// }

// void ManagedMemoryExam() {
//     foo = 42;

//     int *bar;
//     cudaMallocManaged(&bar, 4);
//     *bar = 0xcaffe;

//     kernel<<<1, 1>>>(bar);
//     cudaDeviceSynchronize();
// }

// __managed__ int x = 3, y = 2;

// __global__ void kernel1() {
//     y = 10;
//     printf("x: %d y: %d\n", x, y);
// }

// void ManagedMemoryExam1() {
//     x = 10;
//     kernel1<<<1, 1>>>();
//     // std::this_thread::sleep_for(std::chrono::milliseconds(1));
//     x = 20;
//     cudaDeviceSynchronize();
//     x = 30;
//     printf("after sync: %d %d\n", x, y);
// }

// void ManagedMemoryStreamExam() {
//     cudaStream_t s1;
//     cudaStreamCreate(&s1);

//     cudaStreamAttachMemAsync(s1, &y, 0, cudaMemAttachSingle);
//     kernel1<<<1, 1>>>();

//     printf("y: %d\n", y);
//     y = 20;
//     auto res1 = cudaDeviceSynchronize();
//     if (res1 != cudaSuccess) {
//         std::cerr << "CUDA error: " << cudaGetErrorString(res1) << std::endl;
//     }

//     printf("y: %d\n", y);
// }

__global__ void threadLogic() {
    if (threadIdx.x % 2 == 0) {
        printf("Thread %d is even\n", threadIdx.x);
    } else {
        printf("Thread %d is odd\n", threadIdx.x);
    }
}

void ThreadExam() {

}

int main() {
    // BasicExam();
    // GridExam();
    // takeNTurns<<<1, 1>>>("main", 5);
    // testScheduling<<<1, 4>>>(5);
    // ReduceExam();
    // EventExam();
    // NsightExam();
    // MemoryExam();
    // SharedMemoryExam();
    // ManagedMemoryExam();
    // ManagedMemoryExam1();
    // ManagedMemoryStreamExam();
    ThreadExam();

    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    return 0;
}
