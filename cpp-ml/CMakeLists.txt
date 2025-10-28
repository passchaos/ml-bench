cmake_minimum_required(VERSION 3.30)
project(CPPML LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -O3 -g -march=native -free-vectorize")

# add_executable(learn src/learn.cu src/utility.cuh)
add_executable(matmul src/matmul.cu)
target_compile_options(matmul PRIVATE -Wall -Wextra -O3 -g -march=native)
# target_include_directories(learn PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
