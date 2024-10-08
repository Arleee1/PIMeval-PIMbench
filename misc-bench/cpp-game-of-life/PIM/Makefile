# Makefile: C++ version of game of life
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

PROJ_ROOT = ../../..
include ${PROJ_ROOT}/Makefile.common

SUBDIRS := BitSIMD

.DEFAULT_GOAL := perf

.PHONY: debug perf dramsim3_integ clean $(SUBDIRS)

EXEC := game-of-life.out
SRC := game-of-life.cpp

perf_targets = $(EXEC)

PIM_SIM_TARGET ?= PIM_DEVICE_NONE

ifeq ($(PIM_SIM_TARGET),PIM_DEVICE_BITSIMD_V_AP)
	perf_targets += $(SUBDIRS)
endif

# make USE_OPENMP=1
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
	CXXFLAGS += -fopenmp
endif

debug perf dramsim3_integ: $(perf_targets)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean: $(SUBDIRS)
	rm -rf $(EXEC) *.dSYM

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS) USE_OPENMP=$(USE_OPENMP)