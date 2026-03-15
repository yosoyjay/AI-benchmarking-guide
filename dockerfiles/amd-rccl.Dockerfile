# Dockerfile for AMD RCCL AllReduce bandwidth benchmark.
#
# Builds RCCL and rccl-tests from pinned commits so the Python harness
# can run benchmarks without cloning or compiling at runtime.
#
#   docker build -f amd-rccl.Dockerfile -t ai-bench/amd-rccl .

FROM rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0_triton_llvm_reg_issue

ARG RCCL_COMMIT=57e58688f44c77076ad536ef1f6b68741fc6e694
ARG RCCL_TESTS_COMMIT=40b1b17901370a7880d4a56854b5361c89f8d324

# Build RCCL from source at a pinned commit
RUN git clone https://github.com/ROCm/rccl.git /opt/rccl && \
    cd /opt/rccl && \
    git checkout ${RCCL_COMMIT} && \
    cmake . && make

# Build rccl-tests (standard + MPI variants) against the local RCCL
RUN git clone https://github.com/ROCm/rccl-tests.git /opt/rccl-tests && \
    cd /opt/rccl-tests && \
    git checkout ${RCCL_TESTS_COMMIT} && \
    make HIP_HOME=/opt/rocm NCCL_HOME=/opt/rccl \
         CUSTOM_RCCL_LIB=/opt/rccl/librccl.so && \
    make MPI=1 MPI_HOME=/opt/ompi HIP_HOME=/opt/rocm \
         NCCL_HOME=/opt/rccl
