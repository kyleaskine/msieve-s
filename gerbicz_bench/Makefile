# Standalone Gerbicz collision-search CUDA benchmark.
# Builds a single `bench` executable that runs both CPU and GPU paths
# on identical synthetic data and compares hit counts.

NVCC      ?= nvcc
CXX       ?= g++
CC        ?= gcc

# Target sm_120 native (CUDA 13.2 supports it directly). PTX fallback for older
# arches lets the binary run on the user's other GPUs if any.
NVCC_ARCH ?= -gencode arch=compute_120,code=sm_120 \
             -gencode arch=compute_89,code=compute_89

NVCC_FLAGS = $(NVCC_ARCH) -O3 -std=c++17 -Xcompiler "-O3 -Wall"
CXXFLAGS   = -O3 -Wall -std=c++17
CFLAGS     = -O3 -fomit-frame-pointer -march=native -Wall -Wno-unused-result

LDLIBS    = -lcudart -lm -lgomp

CU_SRCS   = bench_main.cu bench_gpu.cu \
            collision_bucket.cu collision_filter.cu collision_verify.cu
CXX_SRCS  = bench_cpu.c
C_SRCS    = data_gen.c verify_qsort.c bench_cpu_omp.c

CU_OBJS   = $(CU_SRCS:.cu=.o)
CXX_OBJS  = $(CXX_SRCS:.c=.o)
C_OBJS    = $(C_SRCS:.c=.o)
OBJS      = $(CU_OBJS) $(CXX_OBJS) $(C_OBJS)

bench: $(OBJS)
	$(NVCC) $(NVCC_ARCH) -o $@ $^ $(LDLIBS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

bench_cpu.o: bench_cpu.c collision_ref.c bench_cpu.h
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

data_gen.o: data_gen.c data_gen.h
	$(CC) $(CFLAGS) -c $< -o $@

verify_qsort.o: verify_qsort.c verify_qsort.h
	$(CC) $(CFLAGS) -c $< -o $@

bench_cpu_omp.o: bench_cpu_omp.c bench_cpu.h
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@

clean:
	rm -f bench *.o

.PHONY: clean
