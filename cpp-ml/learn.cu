#include <cuda_runtime_api.h>
#include <iostream>

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


int main() {
    deviceInfo();
}
