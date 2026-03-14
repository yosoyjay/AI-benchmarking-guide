
# ND MI300X v5 Benchmark Results

## System Specifications

| GPU           | AMD Instinct MI300X |
|---------------|-------------------|
| CPU           | Intel(R) Xeon(R) Platinum 8480C |
| Ubuntu        |   22.04  |
| ROCm        |   6.8.5  |
| RCCL  | 2.22.3-develop |
| HIP | 6.2.41134 |
| VBIOS  | 113-MI3SRIOV-001 |


## Microbenchmarks
### GEMM HipBLASLt 

The results shown below are with random initialization (best representation of real-life workloads), FP8, and 10,000 warmup iterations.

| m           | n         | k        | ND MI300X v5 (TFLOPS)    |
| ----------- | --------- | -------- | ---------------------- |
| 1024        | 1024      | 1024     | 278.2                   |
| 2048        | 2048      | 2048     |  726.9           |
| 4096        | 4096      | 4096     |  1085.4                |
| 8192        | 8192      | 8192     |  1222.9               |
| 16384       | 16384     | 16384    |  1081.4               |
| 32768       | 32768     | 32768    |  1142.4               |
| \---------- | \-------- | \------- | \--------------------- |
| 1024        | 2145      | 1024     |   396.2                 |
| 6144        | 12288     | 12288    |   1148.7             |
| 802816      | 192       | 768      |    647.4               |

### HBM Bandwidth

|       | ND MI300X v5 (TB/s) |
| ----- | ----------------- |
| Copy  | 4.15              |
| Mul   | 4.28              |
| Add   | 4.13              |
| Triad | 4.10              |
| Dot   | 4.0              |


### Flash Attention 2.0

The performance (in TFLOPS), in table below, represents the performance for a head dimension of 64, a batch size of 2, and a sequence length of 8192.

|       | ND MI300X v5 (TFLOPS) |
| ----- | ----------------- |
| Standard Attention(PyTorch)  | 145.9   |
| Flash Attention 2.0   | 328.6  |

### TransferBench

|                       | ND MI300X v5 (GB/s) |
| --------------------- | ----------------- |
| Host to Device        | 55                |
| Device to Host        | 56                |


### FIO Tests

| Test             | Batch Size(Bytes) | ND MI300X v5 (GB/s) |
| ---------------- | ----------------- | ----------------- |
| Sequential read  | 1M                | 55.9              |
| Sequential read  | 512k              | 57.7              |
| Random read      | 1k                | 1.0              |
| Sequential read  | 1k                | 0.9              |
| Sequential write | 1M                | 52.1              |
| Sequential write | 512k              | 52.7              |
| Random write     | 1k                | 0.3              |
| Sequential write | 1k                | 0.5             |


## RCCL Bandwidth

The values (in GB/s) are the bus bandwidth values obtained from the RCCL AllReduce (NVLS algorithm) tests in-place operations, varying from 1KB to 8GB of data.

| Message Size (Bytes) | ND MI300X v5 (GB/s) |
| -------------------- | ----------------- |
| 1K                   | 0.04              |
| 2K                   | 0.07              |
| 4K                   | 0.15              |
| 8K                   | 0.29              |
| 16K                  | 0.59              |
| 32K                  | 1.13             |
| 65K                  | 2.22              |
| 132K                 | 4.18              |
| 256K                 | 7.50             |
| 524K                 | 13.01             |
| 1M                   | 22.41             |
| 2M                   | 43.41             |
| 4M                   | 82.53            |
| 8M                   | 131.57            |
| 16M                  | 171.46            |
| 33M                  | 230.35            |
| 67M                  | 266.34            |
| 134M                 | 290.81            |
| 268M                 | 302.80            |
| 536M                 | 308.89            |
| 1G                   | 313.92           |
| 2G                   | 315.06            |
| 4G                   | 314.85            |
| 8G                   | 316.82            |

## End-to-End Inference Workloads - vLLM

### LLAMA 3.1 (8B)

Performance results for LLAMA 3.1 (8B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 1       | 128       | 128        | 18149                  |
| 1       | 128       | 1024       | 19579                |
| 1       | 128       | 2048       | 16394                |
| 1       | 500       | 2000       | 13559                 |
| 1       | 1024      | 1024       | 10995                 |
| 1       | 2048      | 2048       | 7356                  |

### LLAMA 3.1 (70B)

Performance results for LLAMA 3.1 (70B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 8       | 128       | 128        | 9026                  |
| 8       | 128       | 1024       | 12461                  |
| 8       | 128       | 2048       | 12204                  |
| 8       | 500       | 2000       | 10877                  |
| 8       | 1024      | 1024       | 8256                   |
| 8       | 2048      | 2048       | 7421                   |

### LLAMA 3 (405B)

Performance results for LLAMA 3 (405B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 8       | 128       | 128        | 2484                   |
| 8       | 128       | 1024       | 3846                   |
| 8       | 128       | 2048       | 3902                   |
| 8       | 500       | 2000       | 3407                  |
| 8       | 1024      | 1024       | 2363                   |
| 8       | 2048      | 2048       | 1840                  |
