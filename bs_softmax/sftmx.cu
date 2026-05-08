#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Dimensions for TinyLlama-style FFN / Attention Scores
#define BATCH_N 1      // 1 token inference
#define MODEL_K 1024   
#define LAYER_M 2048    
#define TILE 16
#define SOFTMAX_THREADS 256

// --- KERNEL 1: TILED GEMM ---
__global__ void gemm_llama_tiled(float* A, float* B, float* C, int n, int m, int k) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE - 1) / TILE; t++) {
        if (row < n && (t * TILE + tx) < k)
            As[ty][tx] = A[row * k + t * TILE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < m && (t * TILE + ty) < k)
            Bs[ty][tx] = B[(t * TILE + ty) * m + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < n && col < m) {
        C[row * m + col] = sum;
    }
}

// --- KERNEL 2: OPTIMIZED FUSED SOFTMAX ---
__global__ void softmax_optimized(float* input, float* output, int m) {
    // Shared memory for finding max and sum
    extern __shared__ float sdata[]; 

    int tid = threadIdx.x;
    float local_max = -1e38f;

    // Step 1: Find Max (Online)
    for (int i = tid; i < m; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Reduction for Max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float final_max = sdata[0];
    __syncthreads();

    // Step 2: Sum Exponentials
    float local_sum = 0.0f;
    for (int i = tid; i < m; i += blockDim.x) {
        local_sum += expf(input[i] - final_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction for Sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float final_sum = sdata[0];

    // Step 3: Normalize and Write
    for (int i = tid; i < m; i += blockDim.x) {
        output[i] = expf(input[i] - final_max) / final_sum;
    }
}

int main() {
    size_t sizeA = BATCH_N * MODEL_K * sizeof(float);
    size_t sizeB = MODEL_K * LAYER_M * sizeof(float);
    size_t sizeC = BATCH_N * LAYER_M * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    float *h_Softmax = (float*)malloc(sizeC);

    // Initialization (Initialize with 1.0 for predictable verification)
    for (int i = 0; i < BATCH_N * MODEL_K; i++) h_A[i] = 1.0f;
    // B initialized so every row is identical for easy checking
    for (int i = 0; i < MODEL_K * LAYER_M; i++) h_B[i] = 0.01f; 

    float *d_A, *d_B, *d_C, *d_Softmax;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMalloc(&d_Softmax, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Launch GEMM
    dim3 threadsGEMM(TILE, TILE);
    dim3 blocksGEMM((LAYER_M + TILE - 1) / TILE, (BATCH_N + TILE - 1) / TILE);
    
    printf("1. Launching Tiled GEMM...\n");
    gemm_llama_tiled<<<blocksGEMM, threadsGEMM>>>(d_A, d_B, d_C, BATCH_N, LAYER_M, MODEL_K);

    // Launch Softmax (Fused)
    printf("2. Launching Optimized Softmax...\n");
    int smem_size = SOFTMAX_THREADS * sizeof(float);
    softmax_optimized<<<BATCH_N, SOFTMAX_THREADS, smem_size>>>(d_C, d_Softmax, LAYER_M);

    // Copy back
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Softmax, d_Softmax, sizeC, cudaMemcpyDeviceToHost);

    // --- VERIFICATION ---
    printf("\n--- Verification Report ---\n");
    printf("Num_threads: %d\n", SOFTMAX_THREADS);
    printf("GEMM Output[0]: %f (Expected: %f)\n", h_C[0], (float)MODEL_K * 0.01f);
    
    float sum_check = 0;
    for(int i=0; i<LAYER_M; i++) sum_check += h_Softmax[i];
    printf("Softmax Probabilities Sum: %f (Expected: 1.000000)\n", sum_check);
    printf("First Prob: %f (Expected: %f)\n", h_Softmax[0], 1.0f/LAYER_M);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_Softmax);
    free(h_A); free(h_B); free(h_C); free(h_Softmax);
    return 0;
}