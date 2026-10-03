# stage23_gpu: GPU size optimization (M1) and root optimization (M2), standalone tools.
#
#   make test               host build of the shared headers + tests vs GMP and CADO
#   make device CUDA=120    compile the device harness for sm_120 (does not run it)
#   make run-device         run the device harness (needs a free GPU)
#   make s23_sopt_cpu       size optimization tool, CPU build (same source as the GPU's)
#   make s23_sopt CUDA=120  size optimization tool, GPU build
#   make accept-cpu         M1 acceptance (13,000 c146 polys vs CADO), CPU build
#   make accept-gpu         M1 acceptance, GPU build (needs a free GPU)
#                           (EFFORT=50 SET=top: effort 50 on the 3000 polys CADO ran at 50;
#                           CPU build only until per-candidate parallelism: on the GPU one
#                           thread would run far past the 60 s WSL2 watchdog)
#
# The device code lives in include/*.h as __host__ __device__ templates, so the host
# tests exercise the same source the GPU runs.

CUDA ?= 120
NVCC ?= nvcc
CXX ?= g++
BUILD := build

# CADO's include flags and libraries, from the same code the stage23_bench tools use
# (polyfmt.cado_compile_args: CADO_BUILD_DIR, else cado_build_dir in ../nfs_config.ini),
# so the tools, the fallback and the acceptance scorer all use one CADO.
# `make CADO_BUILD_DIR=...` overrides it.
POLYFMT := ../stage23_bench/tools/polyfmt.py
ifneq ($(MAKECMDGOALS),clean)
CADO_INC := $(shell python3 $(POLYFMT) --cado-inc "$(CADO_BUILD_DIR)")
CADO_LIBS := $(shell python3 $(POLYFMT) --cado-libs "$(CADO_BUILD_DIR)")
ifeq ($(strip $(CADO_INC)),)
$(error cannot locate CADO (see the message above): set CADO_BUILD_DIR or cado_build_dir in ../nfs_config.ini)
endif
endif
CADO_LIB_FILES := $(filter %.a,$(CADO_LIBS))

HEADERS := $(wildcard include/*.h)
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -Iinclude -Itests
CASES := $(BUILD)/cases.txt

.PHONY: all test device run-device clean s23_sopt_cpu s23_sopt accept-cpu accept-gpu

SOPT_SRC := tools/s23_sopt.cu tools/cado_io.h $(HEADERS)

all: test

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_mpint: tests/test_mpint.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $< -lgmp

$(BUILD)/test_lll: tests/test_lll.cpp $(HEADERS) $(CADO_LIB_FILES) | $(BUILD)
	$(CXX) $(CXXFLAGS) -DS23_TRACK_BITS $(CADO_INC) -o $@ $< $(CADO_LIBS) -fopenmp

$(CASES): tests/gen_cases.py | $(BUILD)
	python3 tests/gen_cases.py $@

test: $(BUILD)/test_mpint $(BUILD)/test_lll $(CASES)
	./$(BUILD)/test_mpint
	./$(BUILD)/test_lll $(CASES)

$(BUILD)/test_device: tests/test_device.cu $(HEADERS) | $(BUILD)
	$(NVCC) -O2 -std=c++17 -arch=sm_$(CUDA) -Iinclude -Itests -o $@ $< -lgmp

device: $(BUILD)/test_device

run-device: $(BUILD)/test_device $(CASES)
	./$(BUILD)/test_device $(CASES) -1

$(BUILD)/cado_io.o: tools/cado_io.c tools/cado_io.h include/sopt_skew.h | $(BUILD)
	gcc -O2 -std=c99 -fopenmp $(CADO_INC) -Itools -Iinclude -c -o $@ $<

# Rounding as CADO: no automatic multiply-add fusion (CADO's C code has none); the fused
# operations of CADO's C++ root finder are explicit fma() calls in include/dpoly.h
$(BUILD)/s23_sopt_cpu: $(SOPT_SRC) $(BUILD)/cado_io.o $(CADO_LIB_FILES)
	$(CXX) -O3 -march=native -ffp-contract=off -std=c++17 -fopenmp -Iinclude -Itools -x c++ tools/s23_sopt.cu \
		-x none $(BUILD)/cado_io.o -o $@ $(CADO_LIBS)

# --fmad=false: no automatic fusion, as in the CPU build (explicit fma() in dpoly.h)
$(BUILD)/s23_sopt: $(SOPT_SRC) $(BUILD)/cado_io.o $(CADO_LIB_FILES)
	$(NVCC) -O2 -std=c++17 -arch=sm_$(CUDA) --fmad=false -Iinclude -Itools -Xcompiler -fopenmp \
		-o $@ tools/s23_sopt.cu $(BUILD)/cado_io.o $(CADO_LIBS) -lgomp

s23_sopt_cpu: $(BUILD)/s23_sopt_cpu
s23_sopt: $(BUILD)/s23_sopt

# M1 acceptance against the c146 fixture (stage23_bench/data/c146/sopt_sample.tsv.gz)
EFFORT ?= 0
SET ?= all
FIXTURE := ../stage23_bench/data/c146/sopt_sample.tsv.gz
RAW := $(BUILD)/fixture_raw_$(SET).poly
ACCEPT_ARGS := $(if $(filter 0,$(EFFORT)),,--effort e$(EFFORT))

$(RAW): $(FIXTURE) | $(BUILD)
	python3 ../stage23_bench/tools/fixture_raw.py $(FIXTURE) $@ --set $(SET)

accept-cpu: $(BUILD)/s23_sopt_cpu $(RAW)
	./$(BUILD)/s23_sopt_cpu -t 8 -sopteffort $(EFFORT) -inputpolys $(RAW) > $(BUILD)/accept_cpu.txt
	cd .. && python3 stage23_bench/tools/sopt_compare.py $(FIXTURE:../%=%) stage23_gpu/$(BUILD)/accept_cpu.txt --rescore $(ACCEPT_ARGS)

accept-gpu: $(BUILD)/s23_sopt $(RAW)
	./$(BUILD)/s23_sopt -sopteffort $(EFFORT) -inputpolys $(RAW) > $(BUILD)/accept_gpu.txt
	cd .. && python3 stage23_bench/tools/sopt_compare.py $(FIXTURE:../%=%) stage23_gpu/$(BUILD)/accept_gpu.txt --rescore $(ACCEPT_ARGS)

clean:
	rm -rf $(BUILD)
