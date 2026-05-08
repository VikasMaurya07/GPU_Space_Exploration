#include <stdio.h>
#include <cuda_runtime.h>

// Dimensions for TinyLlama FFN (1 token inference)
#define BATCH_N 1   // Represents 1 Rows (N)
#define MODEL_K 1024   // Represents 1024 Inner Dimensions (K)
#define LAYER_M 2048   // Represents 2048 Columns (M) 
#define TILE 32

__global__ void gemm_llama_tiled(float* A, float* B, float* C, int n, int m, int k) {
    // Shared memory tiles for A and B
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    // Loop over tiles along the K dimension
    for (int t = 0; t < (k + TILE - 1) / TILE; t++) {
        // Collaborative loading of A tile into Shared Memory
        if (row < n && (t * TILE + tx) < k)
            As[ty][tx] = A[row * k + t * TILE + tx];
        else
            As[ty][tx] = 0.0f;

        // Collaborative loading of B tile into Shared Memory
        if (col < m && (t * TILE + ty) < k)
            Bs[ty][tx] = B[(t * TILE + ty) * m + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads(); // Synchronize to ensure tiles are loaded

        // Compute partial product for this tile
        for (int i = 0; i < TILE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads(); // Synchronize before loading the next tile
    }

    // Write final result to Global Memory
    if (row < n && col < m) {
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

    // Initialization (same as Naive for direct comparison)
    for (int i = 0; i < BATCH_N * MODEL_K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < MODEL_K * LAYER_M; i++) h_B[i] = 0.5f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks((LAYER_M + TILE - 1) / TILE, (BATCH_N + TILE - 1) / TILE);

    printf("Simulating Tiled TinyLlama FFN Layer: [%d x %d] * [%d x %d]\n", BATCH_N, MODEL_K, MODEL_K, LAYER_M);
    
    gemm_llama_tiled<<<blocks, threads>>>(d_A, d_B, d_C, BATCH_N, LAYER_M, MODEL_K);
    
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    printf("C[0] = %f (Expected: %f)\n", h_C[0], (float)MODEL_K * 0.5f);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
