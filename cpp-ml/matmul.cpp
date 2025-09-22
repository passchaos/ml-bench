#include <random>
#include <iostream>

void matmul(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    #define A(i, j) a[(i) * k + (j)]
    #define B(i, j) b[(i) * n + (j)]
    #define C(i, j) c[(i) * n + (j)]

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A(i, p) * B(p, j);
            }

            C(i, j) = sum;
        }
    }

    #undef A
    #undef B
    #undef C
}

void random_mat(int m, int n, float* mat) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<float> standard_normal_dist(0.0, 1.0);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i * n + j] = standard_normal_dist(gen);
        }
    }
}

void copy_mat(int m, int n, const float* src, float* dst) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            dst[i * n + j] = src[i * n + j];
        }
    }
}



void print_mat(int m, int n, const float* mat) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << mat[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}
