cmake_minimum_required(VERSION 3.30)
project(CPPML)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -O3 -g -march=native -free-vectorize")

add_executable(CPPML matmul.cpp)
