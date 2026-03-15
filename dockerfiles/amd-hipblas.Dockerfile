# Dockerfile for AMD hipBLASLt GEMM benchmark.
#
# Builds hipblaslt-bench from a pinned commit so the Python harness can
# run benchmarks without cloning or compiling at runtime.
#
#   docker build -f amd-hipblas.Dockerfile -t ai-bench/amd-hipblas .

FROM rocm/vllm-dev@sha256:9a9582e6161597baeb2d102f2b57fa2cdb24e907d137f4a385f9ce78e505a142

ARG HIPBLASLT_COMMIT=a11ccf64efcd818106dbe37768f69dfcc0a7ff22
ARG GFX_ARCH=gfx942

RUN apt-get update && apt-get -y install llvm-dev && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ROCm/hipBLASLt /opt/hipBLASLt && \
    cd /opt/hipBLASLt && \
    git checkout ${HIPBLASLT_COMMIT} && \
    ./install.sh -dc -a ${GFX_ARCH}
