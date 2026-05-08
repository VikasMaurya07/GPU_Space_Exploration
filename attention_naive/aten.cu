// This naive version explicitly launches separate kernels. 
// This is "naive" because it forces the GPU to write the QK^T matrix (which is NxN) to Global Memory and read it back.

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define BATCH_N 1      
#define SEQ_LEN 1024   // N
#define HEAD_DIM 64   // d
#define TILE 32
#define SOFTMAX_THREADS 256

// --- TILED GEMM (Used for Q*K and S*V) ---
__global__ void gemm_tiled(float* A, float* B, float* C, int n, int m, int k) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty; int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    for (int t = 0; t < (k + TILE - 1) / TILE; t++) {
        if (row < n && (t * TILE + tx) < k) As[ty][tx] = A[row * k + t * TILE + tx];
        else As[ty][tx] = 0.0f;
        if (col < m && (t * TILE + ty) < k) Bs[ty][tx] = B[(t * TILE + ty) * m + col];
        else Bs[ty][tx] = 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) sum += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    if (row < n && col < m) C[row * m + col] = sum;
}

// --- OPTIMIZED SOFTMAX ---
__global__ void softmax_kernel(float* input, float* output, int n, int m) {
    extern __shared__ float sdata[];
    int row = blockIdx.x; int tid = threadIdx.x;
    if (row >= n) return;
    float row_max = -1e38f;
    for (int i = tid; i < m; i += blockDim.x) row_max = fmaxf(row_max, input[row * m + i]);
    sdata[tid] = row_max; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float final_max = sdata[0];
    float row_sum = 0.0f;
    for (int i = tid; i < m; i += blockDim.x) row_sum += expf(input[row * m + i] - final_max);
    sdata[tid] = row_sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float final_sum = sdata[0];
    for (int i = tid; i < m; i += blockDim.x) output[row * m + i] = expf(input[row * m + i] - final_max) / final_sum;
}

int main() {
    // Setup for Q(Nxd), K(Nxd), V(Nxd)
    size_t size_Nd = SEQ_LEN * HEAD_DIM * sizeof(float);
    size_t size_NN = SEQ_LEN * SEQ_LEN * sizeof(float); // Intermediate Attention Matrix!

    float *h_Q, *h_K, *h_V, *h_O;
    h_Q = (float*)malloc(size_Nd); h_K = (float*)malloc(size_Nd);
    h_V = (float*)malloc(size_Nd); h_O = (float*)malloc(size_Nd);

    for(int i=0; i<SEQ_LEN*HEAD_DIM; i++) { h_Q[i] = 1.0f; h_K[i] = 0.01f; h_V[i] = 1.0f; }

    float *d_Q, *d_K, *d_V, *d_S, *d_P, *d_O;
    cudaMalloc(&d_Q, size_Nd); cudaMalloc(&d_K, size_Nd); cudaMalloc(&d_V, size_Nd);
    cudaMalloc(&d_S, size_NN); // Attention scores (Global Memory!)
    cudaMalloc(&d_P, size_NN); // Softmax result (Global Memory!)
    cudaMalloc(&d_O, size_Nd);

    cudaMemcpy(d_Q, h_Q, size_Nd, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size_Nd, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_Nd, cudaMemcpyHostToDevice);

    printf("--- NAIVE ATTENTION BASELINE ---\n");
    // 1. S = Q * K^T
    dim3 threads(TILE, TILE);
    dim3 blocks_NN((SEQ_LEN + TILE - 1)/TILE, (SEQ_LEN + TILE - 1)/TILE);
    gemm_tiled<<<blocks_NN, threads>>>(d_Q, d_K, d_S, SEQ_LEN, SEQ_LEN, HEAD_DIM);

    // 2. P = Softmax(S)
    int smem = SOFTMAX_THREADS * sizeof(float);
    softmax_kernel<<<SEQ_LEN, SOFTMAX_THREADS, smem>>>(d_S, d_P, SEQ_LEN, SEQ_LEN);

    // 3. O = P * V
    dim3 blocks_Nd((HEAD_DIM + TILE - 1)/TILE, (SEQ_LEN + TILE - 1)/TILE);
    gemm_tiled<<<blocks_Nd, threads>>>(d_P, d_V, d_O, SEQ_LEN, HEAD_DIM, SEQ_LEN);

    cudaMemcpy(h_O, d_O, size_Nd, cudaMemcpyDeviceToHost);
    printf("Verification: Output[0] = %f (Expected: ~1.0)\n", h_O[0]);

    return 0;
}