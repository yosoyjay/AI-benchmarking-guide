# Azure ND H200 v5 Benchmark Results

## System Specifications

| GPU           | NVIDIA H200 141GB |
|---------------|-------------------|
| CPU           | Intel(R) Xeon(R) Platinum 8480C |
| Ubuntu        |   22.04  |
| CUDA          |   12.8  |
| NVIDIA Driver | 570.133.20  |
| VBIOS         | 96.00.BC.00.02 |
| NCCL          |    2.26.2  |
| PyTorch       |    2.7.0   |


## Microbenchmarks
### GEMM CuBLASLt 

The results shown below are with random initialization (best representation of real-life workloads), FP8, and 10,000 warmup iterations.

| m           | n         | k        | ND H200 V5 (TFLOPS)    |
| ----------- | --------- | -------- | ---------------------- |
| 1024        | 1024      | 1024     | 403.4                  |
| 2048        | 2048      | 2048     | 1134.6                 |
| 4096        | 4096      | 4096     | 1249.2                 |
| 8192        | 8192      | 8192     | 1269.4                 |
| 16384       | 16384     | 16384    | 1305.0                 |
| 32768       | 32768     | 32768    | 1333.2                |
| \---------- | \-------- | \------- | \--------------------- |
| 1024        | 2145      | 1024     | 648.0                  |
| 802816      | 192       | 768      | 839.8                 |

### HBM Bandwidth

|       | ND H200 V5 (TB/s) |
| ----- | ----------------- |
| Copy  | 4.01              |
| Mul   | 4.00              |
| Add   | 4.19              |
| Triad | 4.18              |
| Dot   | 4.57              |


### Flash Attention 2.0

The performance (in TFLOPS), in table below, represents the performance for a head dimension of 128, a batch size of 2, and a sequence length of 8192.

|       | ND H200 V5 (TFLOPS) |
| ----- | ----------------- |
| Standard Attention(PyTorch)  | 161.5   |
| Flash Attention 2.0   | 329.3  |

### NV Bandwidth

|                       | ND H200 V5 (GB/s) |
| --------------------- | ----------------- |
| Host to Device        | 55                |
| Device to Host        | 55                |
| Device to Device read | 778               |


### FIO Tests

| Test             | Batch Size(Bytes) | ND H200 V5 (GB/s) |
| ---------------- | ----------------- | ----------------- |
| Sequential read  | 1M                | 55.4              |
| Sequential read  | 512k              | 55.6              |
| Random read      | 1k                | 1.26              |
| Sequential read  | 1k                | 1.56              |
| Sequential write | 1M                | 33.8              |
| Sequential write | 512k              | 33.9              |
| Random write     | 1k                | 0.37              |
| Sequential write | 1k                | 0.39              |


## NCCL Bandwidth

The values (in GB/s) are the bus bandwidth values obtained from the NCCL AllReduce (NVLS algorithm) tests in-place operations, varying from 1KB to 8GB of data.

| Message Size (Bytes) | ND H200 V5 (GB/s) |
| -------------------- | ----------------- |
| 1K                   | 0.06              |
| 2K                   | 0.12              |
| 4K                   | 0.25              |
| 8K                   | 0.49              |
| 16K                  | 0.98              |
| 32K                  | 2.03              |
| 65K                  | 3.97              |
| 132K                 | 7.57              |
| 256K                 | 15.10             |
| 524K                 | 30.02             |
| 1M                   | 57.15             |
| 2M                   | 88.73             |
| 4M                   | 137.77            |
| 8M                   | 179.84            |
| 16M                  | 240.37            |
| 33M                  | 298.12            |
| 67M                  | 367.56            |
| 134M                 | 413.01            |
| 268M                 | 434.72            |
| 536M                 | 445.62            |
| 1G                   | 468.11            |
| 2G                   | 474.83            |
| 4G                   | 478.73            |
| 8G                   | 480.94            |

## End-to-End Inference Workloads - TensorRT-LLM



### LLAMA 3.1 (8B)

Performance results for LLAMA 3.1 (8B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 1       | 128       | 128        | 25302                  |
| 1       | 128       | 1024       | 30235                  |
| 1       | 128       | 2048       | 25366                  |
| 1       | 500       | 2000       | 21086                  |
| 1       | 1024      | 1024       | 16828                  |
| 1       | 2048      | 2048       | 11642                  |

### LLAMA 3 (70B)

Performance results for LLAMA 3 (70B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 8       | 128       | 128        | 12306                  |
| 8       | 128       | 1024       | 18583                  |
| 8       | 128       | 2048       | 18573                  |
| 8       | 500       | 2000       | 16109                  |
| 8       | 1024      | 1024       | 11389                  |
| 8       | 2048      | 2048       | 10495                  |

### LLAMA 3 (405B)

Performance results for LLAMA 3 (405B) with FP8 quantization, 1000 requests.

| tp size | input len | output len | throughput(tokens/sec) |
|---------|-----------|------------|------------------------|
| 8       | 128       | 128        | 3262                   |
| 8       | 128       | 1024       | 5082                   |
| 8       | 128       | 2048       | 5179                   |
| 8       | 500       | 2000       | 4530                   |
| 8       | 1024      | 1024       | 3121                   |
| 8       | 2048      | 2048       | 3014                   |

## End-to-End Pretraining Workloads - Single Node
### LLAMA 3 (3B)
<img width="600" height="600" alt="LLAMA3_3b_Pretrain_Results-H200" src="https://github.com/user-attachments/assets/9f900f60-65b7-439d-b58e-86434ab274b2" />


### LLAMA 3 (8B)
<img width="600" height="600" alt="LLAMA3_8b_Pretrain_Results-H200" src="https://github.com/user-attachments/assets/229bbf32-7cba-42f0-90f2-05039226d52d" />

