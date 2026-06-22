// CUDA Graph feasibility probe for the ds4 decode tape.
//
// The decode tape is ~430 small kernel launches/token with gaps between them
// (wd.sh shows the GB10 GPU is idle ~90% of a decode token). This probe
// measures the wall-clock cost of replaying a synthetic tape of many small
// kernels two ways:
//   (A) manual dispatch (<<<>>> per kernel, as ds4 does today)
//   (B) captured CUDA Graph replayed with cudaGraphLaunch
//
// Kernels read their per-replay "dynamic" args (a position-like index) from a
// device-memory args buffer, so the graph can be replayed without per-node
// param updates -- mirroring the planned args-buffer approach for ds4.
//
// Build:  nvcc -O3 -arch=native -o /tmp/graph_probe graph_probe.cu -lcudart
// Run:    /tmp/graph_probe [n_kernels] [n_replays]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

// A small kernel shaped roughly like a decode sub-step: 1 block, 256 threads,
// reads a per-replay index from an args buffer, does a little arithmetic, writes
// a sink. Many of these back-to-back is the decode tape's signature.
__global__ void small_step(float *sink, const uint32_t *args, uint32_t n_args) {
    uint32_t a = args[0];            // "pos"
    uint32_t b = args[1];            // "token"
    if (threadIdx.x == 0) sink[blockIdx.x] = (float)(a ^ b);  // trivial
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    int n_kernels = argc > 1 ? atoi(argv[1]) : 430;
    int n_replays = argc > 2 ? atoi(argv[2]) : 200;
    int blocks_per = 1;  // decode kernels are tiny (few blocks)

    // One sink per kernel, one args buffer (2 uint32 per replay).
    float *sink;      CHK(cudaMalloc(&sink, n_kernels * sizeof(float)));
    uint32_t *args;   CHK(cudaMalloc(&args, 2 * sizeof(uint32_t)));

    // Warm up.
    for (int i = 0; i < 5; i++) {
        small_step<<<blocks_per, 32>>>(sink, args, 2);
    }
    CHK(cudaDeviceSynchronize());

    // (A) manual dispatch.
    double t0 = now_s();
    for (int r = 0; r < n_replays; r++) {
        uint32_t host_args[2] = {(uint32_t)r, (uint32_t)(r + 1)};
        CHK(cudaMemcpyAsync(args, host_args, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
        for (int k = 0; k < n_kernels; k++) {
            small_step<<<blocks_per, 32>>>(sink + k, args, 2);
        }
    }
    CHK(cudaDeviceSynchronize());
    double t_manual = now_s() - t0;

    // (B) capture once, replay n_replays times.
    cudaStream_t stream;  CHK(cudaStreamCreate(&stream));
    cudaGraph_t graph;    cudaGraphExec_t exec;
    CHK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < n_kernels; k++) {
        small_step<<<blocks_per, 32, 0, stream>>>(sink + k, args, 2);
    }
    CHK(cudaStreamEndCapture(stream, &graph));
    CHK(cudaGraphInstantiate(&exec, graph, 0));

    // warm up replay
    for (int i = 0; i < 5; i++) { CHK(cudaGraphLaunch(exec, stream)); }
    CHK(cudaStreamSynchronize(stream));

    double t1 = now_s();
    for (int r = 0; r < n_replays; r++) {
        uint32_t host_args[2] = {(uint32_t)r, (uint32_t)(r + 1)};
        CHK(cudaMemcpyAsync(args, host_args, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        CHK(cudaGraphLaunch(exec, stream));
    }
    CHK(cudaStreamSynchronize(stream));
    double t_graph = now_s() - t1;

    double per_manual = t_manual / n_replays * 1e3;   // ms/token
    double per_graph  = t_graph  / n_replays * 1e3;
    printf("n_kernels=%d n_replays=%d blocks/kernel=%d\n", n_kernels, n_replays, blocks_per);
    printf("manual:  %.3f ms/token  (%.1f us/kernel)\n", per_manual, per_manual*1000/n_kernels);
    printf("graph :  %.3f ms/token  (%.1f us/kernel)\n", per_graph,  per_graph *1000/n_kernels);
    printf("speedup: %.2fx  (saves %.2f ms/token)\n", per_manual/per_graph, per_manual - per_graph);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(sink); cudaFree(args);
    return 0;
}
