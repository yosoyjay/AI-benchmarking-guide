# Azure ND H100 v5 Benchmark Results

## System Specifications

| GPU           | NVIDIA H100 80GB HBM3 |
|---------------|-------------------|
| CPU           | Intel(R) Xeon(R) Platinum 8480C |
| Ubuntu        |   22.04  |
| CUDA          |   12.6  |
| NVIDIA Driver | 560.35.03   |
| VBIOS         | 96.00.9D.00.02 |
| NCCL          |    2.18.5  |
| PyTorch       |    2.5.1   |


## Microbenchmarks
### GEMM CuBLASLt 

The results shown below are with random initialization (best representation of real-life workloads), FP8, and 10,000 warmup iterations.

| m           | n         | k        | ND H100 V5 (TFLOPS)    |
| ----------- | --------- | -------- | ---------------------- |
| 1024        | 1024      | 1024     | 274.1                   |
| 2048        | 2048      | 2048     | 1011.3                |
| 4096        | 4096      | 4096     | 1216.9                 |
| 8192        | 8192      | 8192     | 1290.1                 |
| 16384       | 16384     | 16384    | 1377.3                |
| 32768       | 32768     | 32768    | 1382.2                 |
| \---------- | \-------- | \------- | \--------------------- |
| 1024        | 2145      | 1024     | 410.7                   |
| 6144        | 12288     | 12288    | 1351.4                 |
| 802816      | 192       | 768      | 663.8                  |

### HBM Bandwidth

|       | ND H100 V5 (TB/s) |
| ----- | ----------------- |
| Copy  | 2.90              |
| Mul   | 2.90              |
| Add   | 2.97              |
| Triad | 2.97              |
| Dot   | 3.18              |


### Flash Attention 2.0

The performance (in TFLOPS), in table below, represents the performance for a head dimension of 128, a batch size of 2, and a sequence length of 8192.

|       | ND H100 V5 (TFLOPS) |
| ----- | ----------------- |
| Standard Attention(PyTorch)  | 145.1   |
| Flash Attention 2.0   | 327.9  |

### NV Bandwidth

|                       | ND H100 V5 (GB/s) |
| --------------------- | ----------------- |
| Host to Device        | 51                |
| Device to Host        | 52                |
| Device to Device read | 672               |


### FIO Tests

| Test             | Batch Size(Bytes) | ND H100 V5 (GB/s) |
| ---------------- | ----------------- | ----------------- |
| Sequential read  | 1M                | 55.3              |
| Sequential read  | 512k              | 55.4              |
| Random read      | 1k                | 1.27              |
| Sequential read  | 1k                | 1.61              |
| Sequential write | 1M                | 33.9              |
| Sequential write | 512k              | 33.9              |
| Random write     | 1k                | 0.37              |
| Sequential write | 1k                | 0.39              |


## NCCL Bandwidth

The values (in GB/s) are the bus bandwidth values obtained from the NCCL AllReduce (NVLS algorithm) tests in-place operations, varying from 1KB to 8GB of data.

| Message Size (Bytes) | ND H100 V5 (GB/s) |
| -------------------- | ----------------- |
| 1K                   | 0.07              |
| 2K                   | 0.14              |
| 4K                   | 0.27              |
| 8K                   | 0.53              |
| 16K                  | 0.95              |
| 32K                  | 1.71             |
| 65K                  | 2.72              |
| 132K                 | 5.40              |
| 256K                 | 10.86             |
| 524K                 | 21.35             |
| 1M                   | 43.70             |
| 2M                   | 81.93             |
| 4M                   | 128.73            |
| 8M                   | 172.13            |
| 16M                  | 234.45            |
| 33M                  | 295.53            |
| 67M                  | 361.78            |
| 134M                 | 400.61            |
| 268M                 | 423.38            |
| 536M                 | 438.46            |
| 1G                   | 466.35           |
| 2G                   | 472.36            |
| 4G                   | 475.54            |
| 8G                   | 478.87            |

## End-to-End Inference Workloads - TensorRT-LLM

### LLAMA 3.1 (8B)

Performance results for LLAMA 3.1 (8B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 1       | 128       | 128        | 15163                  |
| 1       | 128       | 1024       | 17075                  |
| 1       | 128       | 2048       | 15014                  |
| 1       | 500       | 2000       | 13047                  |
| 1       | 1024      | 1024       | 11724                  |
| 1       | 2048      | 2048       | 8026                   |

### LLAMA 3.1 (70B)

Performance results for LLAMA 3.1 (70B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 8       | 128       | 128        | 10032                  |
| 8       | 128       | 1024       | 13345                  |
| 8       | 128       | 2048       | 12910                  |
| 8       | 500       | 2000       | 11594                  |
| 8       | 1024      | 1024       | 9125                   |
| 8       | 2048      | 2048       | 7908                   |

### LLAMA 3 (405B)

Performance results for LLAMA 3 (405B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 8       | 128       | 128        | 2515                   |
| 8       | 128       | 1024       | 3417                   |
| 8       | 128       | 2048       | 3434                   |
| 8       | 500       | 2000       | 3143                   |
| 8       | 1024      | 1024       | 2422                   |
| 8       | 2048      | 2048       | 1973                   |
