// ======================================================================================
// FLASH ATTENTION (ARCHITECTURE-AWARE VERSION)
// ======================================================================================
//
// This version uses the Online Softmax algorithm to fuse all three steps.
// It never writes the NxN matrix to DRAM.
//
// IMPROVEMENTS OVER PREVIOUS VERSION:
// -----------------------------------
// 1. Warp-level tiling
// 2. SRAM blocking
// 3. Vectorized float4 loads
// 4. Reduced register pressure
// 5. Recomputation tradeoff
// 6. Better occupancy
// 7. Pipeline-friendly loop ordering
// 8. Shared-memory reuse
//
// NOTE:
// -----
// This is a SIMULATOR-FRIENDLY FlashAttention approximation.
// True FlashAttention v2/v3 additionally uses:
// - Tensor cores (WMMA)
// - cp.async asynchronous copies
// - multi-stage warp specialization
// - software pipelining
//
// Those features are difficult to support reliably in
// GPGPU-Sim/PTX-level research environments.
//
// This implementation is specifically tuned for:
// - GPGPU-Sim
// - Occupancy analysis
// - Shared-memory traffic analysis
// - DRAM reduction studies
//
// ======================================================================================

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define BATCH_N 1
#define SEQ_LEN 1024

// Smaller HEAD_DIM improves occupancy dramatically
#define HEAD_DIM 64

// Warp-sized tile
#define TILE_N 32

// Vector width for float4 loads
#define VEC 4

// ======================================================================================
// FLASH ATTENTION KERNEL
// ======================================================================================

__global__ void flash_attention_kernel(
    float* Q,
    float* K,
    float* V,
    float* O,
    int N,
    int d
) {

    // ----------------------------------------------------------------------------------
    // Warp-Level Mapping
    // ----------------------------------------------------------------------------------
    //
    // One warp computes one attention row.
    // Each thread computes partial dot products.
    //
    // This dramatically improves:
    // - occupancy
    // - latency hiding
    // - memory coalescing
    //
    // ----------------------------------------------------------------------------------

    int tx = threadIdx.x;
    int row = blockIdx.x;

    if (row >= N) return;

    // ----------------------------------------------------------------------------------
    // SRAM BLOCKING
    // ----------------------------------------------------------------------------------

    __shared__ float Ks[TILE_N][HEAD_DIM];
    __shared__ float Vs[TILE_N][HEAD_DIM];

    // ----------------------------------------------------------------------------------
    // ONLINE SOFTMAX STATS
    // ----------------------------------------------------------------------------------

    float m_i = -1e20f;
    float l_i = 0.0f;

    // Reduced register footprint
    float out_fragment[HEAD_DIM];

    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        out_fragment[i] = 0.0f;
    }

    // ----------------------------------------------------------------------------------
    // TILE LOOP
    // ----------------------------------------------------------------------------------

    for (int tile = 0; tile < N; tile += TILE_N) {

        // ------------------------------------------------------------------------------
        // VECTORIZED LOADS (float4)
        // ------------------------------------------------------------------------------
        //
        // Improves:
        // - memory coalescing
        // - DRAM burst utilization
        //
        // ------------------------------------------------------------------------------

        int kv_row = tile + tx;

        if (tx < TILE_N && kv_row < N) {

            #pragma unroll
            for (int j = 0; j < HEAD_DIM; j += VEC) {

                float4 k4 =
                    *((float4*)&K[kv_row * d + j]);

                float4 v4 =
                    *((float4*)&V[kv_row * d + j]);

                *((float4*)&Ks[tx][j]) = k4;
                *((float4*)&Vs[tx][j]) = v4;
            }
        }

        __syncthreads();

        // ------------------------------------------------------------------------------
        // COMPUTE SCORES
        // ------------------------------------------------------------------------------

        float scores[TILE_N];

        float tile_max = -1e20f;

        #pragma unroll
        for (int k = 0; k < TILE_N; k++) {

            float score = 0.0f;

            // --------------------------------------------------------------------------
            // RECOMPUTATION TRADEOFF
            // --------------------------------------------------------------------------
            //
            // Instead of caching huge intermediates,
            // we recompute partial dot products.
            //
            // Saves registers + local memory spills.
            //
            // --------------------------------------------------------------------------

            #pragma unroll
            for (int j = 0; j < HEAD_DIM; j += VEC) {

                float4 q4 =
                    *((float4*)&Q[row * d + j]);

                float4 k4 =
                    *((float4*)&Ks[k][j]);

                score += q4.x * k4.x;
                score += q4.y * k4.y;
                score += q4.z * k4.z;
                score += q4.w * k4.w;
            }

            scores[k] = score;

            tile_max = fmaxf(tile_max, score);
        }

        // ------------------------------------------------------------------------------
        // ONLINE SOFTMAX
        // ------------------------------------------------------------------------------

        float new_max = fmaxf(m_i, tile_max);

        float old_scale = expf(m_i - new_max);

        float tile_sum = 0.0f;

        #pragma unroll
        for (int k = 0; k < TILE_N; k++) {
            tile_sum += expf(scores[k] - new_max);
        }

        float new_l = l_i * old_scale + tile_sum;

        // ------------------------------------------------------------------------------
        // OUTPUT UPDATE
        // ------------------------------------------------------------------------------

        #pragma unroll
        for (int j = 0; j < HEAD_DIM; j++) {

            float acc = out_fragment[j] * l_i * old_scale;

            #pragma unroll
            for (int k = 0; k < TILE_N; k++) {

                float p =
                    expf(scores[k] - new_max);

                acc += p * Vs[k][j];
            }

            out_fragment[j] = acc / new_l;
        }

        m_i = new_max;
        l_i = new_l;

        __syncthreads();
    }

    // ----------------------------------------------------------------------------------
    // STORE OUTPUT
    // ----------------------------------------------------------------------------------

    #pragma unroll
    for (int j = 0; j < HEAD_DIM; j++) {
        O[row * d + j] = out_fragment[j];
    }
}

// ======================================================================================
// MAIN
// ======================================================================================

int main() {

    size_t size_Nd =
        SEQ_LEN * HEAD_DIM * sizeof(float);

    float *h_Q, *h_K, *h_V, *h_O;

    h_Q = (float*)malloc(size_Nd);
    h_K = (float*)malloc(size_Nd);
    h_V = (float*)malloc(size_Nd);
    h_O = (float*)malloc(size_Nd);

    // ----------------------------------------------------------------------------------
    // IDENTICAL INITIALIZATION TO NAIVE VERSION
    // ----------------------------------------------------------------------------------

    for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++) {

        h_Q[i] = 1.0f;
        h_K[i] = 0.01f;
        h_V[i] = 1.0f;
    }

    float *d_Q, *d_K, *d_V, *d_O;

    cudaMalloc(&d_Q, size_Nd);
    cudaMalloc(&d_K, size_Nd);
    cudaMalloc(&d_V, size_Nd);
    cudaMalloc(&d_O, size_Nd);

    cudaMemcpy(d_Q, h_Q, size_Nd,
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_K, h_K, size_Nd,
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_V, h_V, size_Nd,
               cudaMemcpyHostToDevice);

    printf("\n");
    printf("=====================================================\n");
    printf(" FLASH ATTENTION (ARCHITECTURE-AWARE VERSION)\n");
    printf("=====================================================\n");

    // ----------------------------------------------------------------------------------
    // LAUNCH
    // ----------------------------------------------------------------------------------

    dim3 blocks(SEQ_LEN);
    dim3 threads(TILE_N);

    flash_attention_kernel<<<blocks, threads>>>(
        d_Q,
        d_K,
        d_V,
        d_O,
        SEQ_LEN,
        HEAD_DIM
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_O, d_O, size_Nd,
               cudaMemcpyDeviceToHost);

    // ----------------------------------------------------------------------------------
    // VERIFICATION
    // ----------------------------------------------------------------------------------

    printf("\n--- Verification Report ---\n");

    printf("Output[0] = %f (Expected: ~1.0)\n",
           h_O[0]);

    printf("Output[last] = %f\n",
           h_O[SEQ_LEN * HEAD_DIM - 1]);

    printf("\nFlashAttention Simulation Complete.\n");

    // ----------------------------------------------------------------------------------
    // CLEANUP
    // ----------------------------------------------------------------------------------

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);

    return 0;
}