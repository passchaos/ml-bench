#ifndef UTIL_H
#define UTIL_H

namespace util {
    __device__ void WasteTime(unsigned long long duration)
    {
        const unsigned long long start = clock64();
        while ((clock64() - start) < duration);
    }
}

#endif
