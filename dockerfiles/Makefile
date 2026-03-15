IMAGE_PREFIX ?= ai-bench

.PHONY: all hipblas rccl

all: hipblas rccl

hipblas:
	docker build --progress=plain \
		-f amd-hipblas.Dockerfile \
		-t $(IMAGE_PREFIX)/amd-hipblas:latest .

rccl:
	docker build --progress=plain \
		-f amd-rccl.Dockerfile \
		-t $(IMAGE_PREFIX)/amd-rccl:latest .
