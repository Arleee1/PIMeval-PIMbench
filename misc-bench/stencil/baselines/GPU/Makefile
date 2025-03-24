# Makefile: Cuda version of stencil
# Copyright (c) 2025 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

# Compiler settings
NVCC := nvcc
CUDA_DIR ?= $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INCLUDE := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64

ARCH := sm_75
ARCH_FLAG := -arch=$(ARCH)

# Target executable
EXEC := stencil.out

# Source files
SOURCES := stencil.cu

# Compiler flags
CFLAGS := -O3
CXXFLAGS := -std=c++17

# make USE_OPENMP=1
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
	CXXFLAGS += -Xcompiler -fopenmp
endif

# Default target
all: $(EXEC)

# Rule to build the target executable
$(EXEC): $(SOURCES)
	$(NVCC) $(SOURCES) -I$(CUDA_INCLUDE) $(ARCH_FLAG) -L$(CUDA_LIB_DIR) $(CFLAGS) $(CXXFLAGS) -o $(EXEC)

# Rule to clean the project
clean:
	rm -f $(EXEC)
