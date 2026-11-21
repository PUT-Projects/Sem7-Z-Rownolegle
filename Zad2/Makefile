# Generic CUDA Makefile for the lab machines (gpu / gpu1, CUDA 12.5, GTX 1080)
# and any Linux box with CUDA <= 12.x.
#
#   make SRC=cuda_pliki/matrixMul.cu          # -arch=sm_61 (GTX 1080, as in lab instructions)
#   make SRC=cuda_pliki/kernel.cu ARCH=sm_89  # other card, e.g. RTX 4060
#   make SRC=moj.cu dbg=1                     # -G -g device debug
#   make SRC=moj.cu run ARGS="1024 256 4"
#   make clean
#
# Output: build/<name>   (e.g. build/matrixMul)

# /usr/local/cuda on gpu1 is CUDA 13.5 (no sm_61 support) -> prefer the newest 12.x.
CUDA_PATH ?= $(firstword $(wildcard /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.5) /usr/local/cuda)
NVCC     ?= $(CUDA_PATH)/bin/nvcc

SRC  ?= cuda_pliki/matrixMul.cu
NAME := $(basename $(notdir $(SRC)))
OUT  := build/$(NAME)

# sm_61 = GTX 1080 (Pascal), sm_89 = RTX 4060 (Ada). CUDA 13+ cannot build sm_61.
ARCH ?= sm_61
GENCODE := -arch=$(ARCH)

# Host code: OpenMP + AVX2/FMA etc. of the build machine (i7-8700 on gpu/gpu1 has AVX2).
# Build on the machine you run on (gpu/gpu1), not on polluks.
NVCCFLAGS := -std=c++17 -Wno-deprecated-gpu-targets -Xcompiler -fopenmp -Xcompiler -march=native
ifeq ($(dbg),1)
    NVCCFLAGS += -G -g
else
    NVCCFLAGS += -O3 -lineinfo
endif

INCLUDES := -Icuda_pliki -I$(dir $(SRC))

.PHONY: all run archs clean
all: $(OUT)

$(OUT): $(SRC) $(wildcard cuda_pliki/*.h $(dir $(SRC))*.h $(dir $(SRC))*.cuh) | build
	$(NVCC) $(NVCCFLAGS) $(GENCODE) $(INCLUDES) $< -o $@

build:
	mkdir -p build

run: $(OUT)
	./$(OUT) $(ARGS)

archs: $(OUT)
	$(CUDA_PATH)/bin/cuobjdump --list-elf --list-ptx $(OUT)

clean:
	rm -rf build
