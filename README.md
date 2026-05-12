# GPU Space Exploration using Transformer-Inspired CUDA Kernels

---

## Overview

This project performs architectural exploration of transformer-inspired GPU workloads using cycle-level GPU simulation. The work focuses on understanding how transformer kernels interact with GPU microarchitectural components such as compute units, memory hierarchy, warp schedulers, and shared-memory organization.

The study uses:
- CUDA kernel implementations
- GPGPU-Sim / Accel-Sim
- PTX-level simulation
- Transformer-inspired GEMM and Softmax kernels

The project evaluates how architectural parameters affect execution behavior, throughput, memory utilization, occupancy, and synchronization efficiency.

---

## Objectives

- Analyze GPU architectural behavior for transformer-style kernels
- Compare naive and tiled GEMM implementations
- Study memory-bound vs compute-bound execution
- Evaluate shared-memory tiling efficiency
- Analyze reduction-heavy softmax workloads

Explore the effect of:
- Compute density
- Memory bandwidth
- Warp scheduling
- Cache hierarchy
- Shared-memory banks
---

## Kernels Implemented

1. GEMM (Matrix Multiplication)

Transformer Feed-Forward Network inspired matrix multiplication:

(1 x 1024) X (1024 x 2048)

Variants:

- Naive GEMM
- Shared-memory tiled GEMM
-- Tile-8
-- Tile-16
-- Tile-32 configurations
---

2. Softmax Kernel

Reduction-heavy transformer softmax implementation used to analyze:
- Warp scheduling behavior
- Shared-memory bank conflicts
- Cache behavior
- Synchronization overhead
---

## GPU Simulator Used
GPGPU-Sim / Accel-Sim

This project uses:
- PTX-level cycle-accurate simulation
- Configurable GPU architecture modeling
- Warp scheduler simulation
- Cache hierarchy exploration
- Shared-memory modeling

Why simulation was used:

- Architectural exploration without physical GPUs
- Controlled parameter variation
- Fine-grained hardware metrics
- Analysis of bottlenecks in transformer workloads
---

## Architectural Parameters Explored

Parameter	Purpose

- Compute Density	Study scaling with SM count
- Memory Bandwidth	Analyze DRAM bottlenecks
- Warp Scheduler	Compare LRR, GTO, Two-Level
- Shared Memory Banks	Evaluate bank conflicts
- On-Chip Storage	Study cache/shared-memory locality
-L2 Cache Size	Analyze cache hierarchy effects
---

## Metrics Collected

- gpu_tot_sim_cycle	Total execution cycles
- gpu_tot_ipc	Instructions per cycle
- gpu_tot_occupancy	Warp occupancy
- L2_BW_total	L2 bandwidth utilization
- L1D_total_cache_miss_rate	L1 cache miss rate
- gpgpu_n_mem_read_global	Global memory reads
- gpgpu_n_shmem_bkconflict	Shared-memory bank conflicts
---

## Key Findings

GEMM Kernel

Tiled GEMM significantly outperformed naive GEMM. Tile-16 achieved the best balance between:
throughput
occupancy
synchronization overhead

Increasing compute density reduced execution cycles initially

Performance eventually saturated due to memory limitations

Larger tile sizes increased shared-memory bank conflicts
---

Softmax Kernel

Softmax behavior differed significantly from GEMM

Performance depended heavily on:

- synchronization
- warp scheduling
- shared-memory organization


GTO scheduling produced higher IPC than LRR

Increasing shared-memory banks dramatically reduced bank conflicts

L2 cache size had minimal effect on memory reads

---

Major Conclusions

- Transformer workloads stress multiple GPU subsystems differently
- GEMM is dominated by:
- memory locality
- arithmetic intensity
- shared-memory reuse

Softmax is dominated by:
- synchronization
- reduction behavior
- warp scheduling efficiency

GPU simulation provides valuable insight into:

- compute-bound behavior
- memory-bound behavior
- synchronization bottlenecks
---

Related Work

This project draws inspiration from:

Transformer architecture research

FlashAttention

GPU architectural simulation

Transformer workload optimization


## Important references:

GPGPU-Sim

FlashAttention Paper

FlashAttention-2

Kernels were taken from the CUDA Programming Guide 
