#include <stdio.h>
#include <cuda_runtime.h>

// Dimensions for TinyLlama FFN (1 token inference)
#define BATCH_N 1   // Represents 1 Rows (N)
#define MODEL_K 1024   // Represents 1024 Inner Dimensions (K)
#define LAYER_M 2048   // Represents 2048 Columns (M)

__global__ void gemm_llama_naive(float* A, float* B, float* C, int n, int m, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < m) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * m + col];
        }
        C[row * m + col] = sum;
    }
}

int main() {
    size_t sizeA = BATCH_N * MODEL_K * sizeof(float);
    size_t sizeB = MODEL_K * LAYER_M * sizeof(float);
    size_t sizeC = BATCH_N * LAYER_M * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // Initialize with data to avoid 0.0 results
    for (int i = 0; i < BATCH_N * MODEL_K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < MODEL_K * LAYER_M; i++) h_B[i] = 0.5f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Grid config: 1x352 blocks (for 5632 columns / 16)
    dim3 threads(16, 16);
    dim3 blocks((LAYER_M + threads.x - 1) / threads.x, (BATCH_N + threads.y - 1) / threads.y);

    printf("Simulating TinyLlama FFN Layer: [%d x %d] * [%d x %d]\n", BATCH_N, MODEL_K, MODEL_K, LAYER_M);
    
    gemm_llama_naive<<<blocks, threads>>>(d_A, d_B, d_C, BATCH_N, LAYER_M, MODEL_K);
    
    // Explicitly copy back to ensure simulator populates h_C
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    printf("C[0] = %f (Expected: %f)\n", h_C[0], (float)MODEL_K * 0.5f);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
