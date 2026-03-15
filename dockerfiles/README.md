# Benchmark Docker Images

Dockerfiles for benchmarks that need to compile software.  Building at
image-creation time (rather than at benchmark-run time) gives us:

- **Reproducibility** -- pinned base images and commit SHAs mean every
  build produces the same binaries.
- **Speed** -- Docker layer caching avoids redundant compiles.
- **Auditability** -- the full build recipe is version-controlled.

Benchmarks that already use pre-built vendor images (`llm_benchmark`,
`flash_attention`, `llama3_run`) do not need Dockerfiles here.

## Building

```bash
cd dockerfiles
make all            # builds every image
make hipblas        # just hipBLASLt
make rccl           # just RCCL
```

Override the image prefix for a custom registry:

```bash
IMAGE_PREFIX=myregistry.azurecr.io/ai-bench make all
```

## Image-to-benchmark mapping

| Dockerfile | Image tag | Benchmark module |
|---|---|---|
| `amd-hipblas.Dockerfile` | `ai-bench/amd-hipblas:latest` | `benchmarks.amd.gemm_hipblas_lt` |
| `amd-rccl.Dockerfile` | `ai-bench/amd-rccl:latest` | `benchmarks.amd.rccl_bandwidth` |

## Updating pinned commits

Each Dockerfile declares the source commit as a build `ARG`:

```dockerfile
ARG HIPBLASLT_COMMIT=a11ccf64...
```

To update, edit the `ARG` default value to the new commit SHA and
rebuild with `make`.  The base image digest can be updated similarly by
changing the `FROM` line.
