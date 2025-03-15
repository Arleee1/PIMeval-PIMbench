# Makefile: Cuda version of stencil
# Copyright (c) 2025 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

# Compiler settings
NVCC := nvcc
CUDA_DIR ?= $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INCLUDE := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64

ARCH := sm_80
ARCH_FLAG := -arch=$(ARCH)

# Target executable
EXEC := stencil.out

# Source files
SOURCES := stencil.cu

# cuSten flags
cuSten_ROOT := ./cuSten/cuSten
cuSten_LIB_DIR := $(cuSten_ROOT)/lib
cuSten_LIB := $(cuSten_LIB_DIR)/libcuSten.a
cuSten_FLAGS := -I$(cuSten_ROOT) -L$(cuSten_LIB_DIR) -lcuSten

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
$(EXEC): $(SOURCES) $(cuSten_LIB)
	$(NVCC) $(SOURCES) $(cuSten_FLAGS) -I$(CUDA_INCLUDE) $(ARCH_FLAG) -L$(CUDA_LIB_DIR) $(FOPENMP) $(CFLAGS) $(CXXFLAGS) $(INCLUDE_FLAGS) -o $(EXEC)

# Rule to build cuSten dependency
$(cuSten_LIB):
	$(MAKE) -C $(cuSten_ROOT) ARCH=$(ARCH)

# Rule to clean the project
clean:
	rm -f $(EXEC)
	$(MAKE) -C $(cuSten_ROOT) clean
