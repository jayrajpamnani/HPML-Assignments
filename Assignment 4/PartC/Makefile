# CUDA Path Configuration
# Allow overriding CUDA_PATH from environment (useful for HPC clusters)
CUDA_PATH ?= /usr/local/cuda
CUDA_COMPILER := $(CUDA_PATH)/bin/nvcc
CUDA_LIB_PATH := -L$(CUDA_PATH)/lib64
CUDA_INCLUDE_PATH := -I$(CUDA_PATH)/include
CUDNN_OPTIONS := -lcudnn -lcudart  # Added -lcudart which is required

# Project Configuration
ARCHIVE_NAME := jmp10051_makefile.tar
EXECUTABLES := convl

# Compilation Flags
# Allow overriding GPU architecture from environment (useful for different GPU types)
GPU_ARCH ?= sm_70
NVCC_FLAGS := -O3 -arch=$(GPU_ARCH)  # Added optimization and architecture flags

# Default Target
all: $(EXECUTABLES)

# Main Compilation Rule
convl: convolve.cu
	$(CUDA_COMPILER) $(NVCC_FLAGS) $(CUDA_INCLUDE_PATH) -o $@ $< $(CUDA_LIB_PATH) $(CUDNN_OPTIONS)

# Clean Target
clean:
	rm -f $(EXECUTABLES) *.o

# Archive Target
tar:
	tar -cvf $(ARCHIVE_NAME) Makefile *.h *.cu *.pdf *.txt *.cpp

# Additional Target (if needed)
partc: partc.cu
	$(CUDA_COMPILER) $(NVCC_FLAGS) $(CUDA_INCLUDE_PATH) -o $@ $< $(CUDA_LIB_PATH) $(CUDNN_OPTIONS)

.PHONY: all clean tar

