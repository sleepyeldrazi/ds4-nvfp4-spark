CC ?= cc

# CPU native flag. The engine is CUDA-only now (Metal support was removed); this
# Makefile targets the Linux/CUDA build path.
NATIVE_CPU_FLAG ?= -march=native

# Include paths for the reorganized src/ layout.
SRC_INCLUDES = -Isrc/core -Isrc/gpu -Isrc/vendor/linenoise -Isrc/vendor/rax

CFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only $(SRC_INCLUDES)

LDLIBS ?= -lm -pthread

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) $(SRC_INCLUDES) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread $(addprefix -Xcompiler ,$(SRC_INCLUDES))
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas

# GPU build links ds4.o (engine) + ds4_cuda.o (CUDA backend implementing
# ds4_gpu.h) + ds4_util.o (shared engine utilities). CPU-only reference build
# links ds4_cpu.o + ds4_util_cpu.o (both compiled with -DDS4_NO_GPU) and skips
# the backend.
CUDA_OBJS = ds4_cuda.o ds4_turbo4.o
CORE_OBJS = ds4.o ds4_util.o ds4_gguf.o ds4_quant.o ds4_tokenizer.o ds4_weights.o $(CUDA_OBJS)
CPU_CORE_OBJS = ds4_cpu.o ds4_util_cpu.o ds4_gguf_cpu.o ds4_quant_cpu.o ds4_tokenizer_cpu.o ds4_weights_cpu.o

.PHONY: all clean test cpu cuda-regression

all: ds4 ds4-server ds4-bench

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

# ---- engine core ----
ds4.o: src/core/ds4.c src/core/ds4.h src/core/ds4_internal.h src/gpu/ds4_gpu.h src/core/ds4_cpu.inc src/core/ds4_gpu.inc src/core/ds4_session.inc
	$(CC) $(CFLAGS) -c -o $@ src/core/ds4.c

# ---- model ----
ds4_gguf.o: src/model/ds4_gguf.c src/core/ds4.h src/core/ds4_internal.h src/gpu/ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ src/model/ds4_gguf.c

ds4_weights.o: src/model/ds4_weights.c src/core/ds4.h src/core/ds4_internal.h
	$(CC) $(CFLAGS) -c -o $@ src/model/ds4_weights.c

# ---- tokenizer ----
ds4_tokenizer.o: src/tokenizer/ds4_tokenizer.c src/core/ds4.h src/core/ds4_internal.h
	$(CC) $(CFLAGS) -c -o $@ src/tokenizer/ds4_tokenizer.c

# ---- quant ----
ds4_quant.o: src/quant/ds4_quant.c src/core/ds4.h src/core/ds4_internal.h
	$(CC) $(CFLAGS) -c -o $@ src/quant/ds4_quant.c

# ---- util ----
ds4_util.o: src/util/ds4_util.c src/core/ds4.h src/core/ds4_internal.h
	$(CC) $(CFLAGS) -c -o $@ src/util/ds4_util.c

# ---- cli ----
ds4_cli.o: src/cli/ds4_cli.c src/core/ds4.h src/vendor/linenoise/linenoise.h
	$(CC) $(CFLAGS) -c -o $@ src/cli/ds4_cli.c

# ---- server ----
ds4_server.o: src/server/ds4_server.c src/core/ds4.h src/vendor/rax/rax.h
	$(CC) $(CFLAGS) -c -o $@ src/server/ds4_server.c

# ---- bench ----
ds4_bench.o: src/bench/ds4_bench.c src/core/ds4.h
	$(CC) $(CFLAGS) -c -o $@ src/bench/ds4_bench.c

# ---- vendor ----
linenoise.o: src/vendor/linenoise/linenoise.c src/vendor/linenoise/linenoise.h
	$(CC) $(CFLAGS) -c -o $@ src/vendor/linenoise/linenoise.c

rax.o: src/vendor/rax/rax.c src/vendor/rax/rax.h src/vendor/rax/rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ src/vendor/rax/rax.c

# ---- gpu (CUDA) ----
ds4_cuda.o: src/gpu/ds4_cuda.cu src/gpu/ds4_cuda_common.h src/gpu/ds4_iq2_tables_cuda.inc src/gpu/ds4_cuda_embed.cuh src/gpu/ds4_cuda_matmul.cuh src/gpu/ds4_cuda_devutil.cuh src/gpu/ds4_cuda_q8.cuh src/gpu/ds4_cuda_norm.cuh src/gpu/ds4_cuda_rope.cuh src/gpu/ds4_cuda_attention.cuh src/gpu/ds4_cuda_hc.cuh src/gpu/ds4_cuda_compressor.cuh src/gpu/ds4_cuda_router.cuh src/gpu/ds4_cuda_indexer.cuh src/gpu/ds4_cuda_dispatch1.cuh src/gpu/ds4_cuda_dispatch2.cuh src/gpu/ds4_cuda_dispatch3.cuh src/gpu/ds4_cuda_quant.cuh src/gpu/ds4_cuda_moe.cuh src/gpu/ds4_cuda_moe_dispatch.cuh src/gpu/ds4_cuda_dispatch4.cuh
	$(NVCC) $(NVCCFLAGS) -c -o $@ src/gpu/ds4_cuda.cu

ds4_turbo4.o: src/gpu/ds4_turbo4.cu src/gpu/ds4_cuda_common.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ src/gpu/ds4_turbo4.cu

# ---- cpu-only reference (compiled with -DDS4_NO_GPU) ----
ds4_cpu.o: src/core/ds4.c src/core/ds4.h src/core/ds4_internal.h src/gpu/ds4_gpu.h src/core/ds4_cpu.inc src/core/ds4_gpu.inc src/core/ds4_session.inc
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/core/ds4.c

ds4_util_cpu.o: src/util/ds4_util.c src/core/ds4.h src/core/ds4_internal.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/util/ds4_util.c

ds4_gguf_cpu.o: src/model/ds4_gguf.c src/core/ds4.h src/core/ds4_internal.h src/gpu/ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/model/ds4_gguf.c

ds4_quant_cpu.o: src/quant/ds4_quant.c src/core/ds4.h src/core/ds4_internal.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/quant/ds4_quant.c

ds4_tokenizer_cpu.o: src/tokenizer/ds4_tokenizer.c src/core/ds4.h src/core/ds4_internal.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/tokenizer/ds4_tokenizer.c

ds4_weights_cpu.o: src/model/ds4_weights.c src/core/ds4.h src/core/ds4_internal.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/model/ds4_weights.c

ds4_cli_cpu.o: src/cli/ds4_cli.c src/core/ds4.h src/vendor/linenoise/linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/cli/ds4_cli.c

ds4_server_cpu.o: src/server/ds4_server.c src/core/ds4.h src/vendor/rax/rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/server/ds4_server.c

ds4_bench_cpu.o: src/bench/ds4_bench.c src/core/ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ src/bench/ds4_bench.c

# ---- tests ----
ds4_test.o: tests/ds4_test.c src/server/ds4_server.c src/core/ds4.h src/vendor/rax/rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c src/gpu/ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ tests/cuda_long_context_smoke.c

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o $(CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: ds4_test.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4_cpu ds4_native ds4_server_test ds4_test *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
