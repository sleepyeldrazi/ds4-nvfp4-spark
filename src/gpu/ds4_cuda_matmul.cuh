__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    if (threadIdx.x == 0) out[tok * out_dim + row] = block_sum_f32(sum);
}

__global__ static void matmul_f16_serial_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok || threadIdx.x != 0) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = 0; i < in_dim; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    out[tok * out_dim + row] = sum;
}

__global__ static void matmul_f16_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    /* Pair loads: __half2 for the F16 weights and float2 for the F32
     * activation, which halves the number of load instructions.  Handles
     * odd chunk boundaries by peeling a single element at each end. */
    uint64_t kk0 = k0, kk1 = k1;
    if ((kk0 & 1u) && kk0 < kk1) {
        sum += __half2float(wr[kk0]) * xr[kk0];
        kk0++;
    }
    const __half2 *wr2 = (const __half2 *)(wr + kk0);
    const float2  *xr2 = (const float2  *)(xr + kk0);
    const uint64_t n2 = (kk1 - kk0) / 2u;
    for (uint64_t i = 0; i < n2; i++) {
        float2 wf = __half22float2(wr2[i]);
        float2 xf = xr2[i];
        sum += wf.x * xf.x;
        sum += wf.y * xf.y;
    }
    if ((kk1 - kk0) & 1u) {
        uint64_t i = kk1 - 1u;
        sum += __half2float(wr[i]) * xr[i];
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[tok * out_dim + row] = total;
    }
}

__global__ static void matmul_f16_pair_ordered_chunks_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;

    __shared__ float partial0[32];
    __shared__ float partial1[32];
    const uint32_t tid = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;
    uint64_t kk0 = k0, kk1 = k1;
    if ((kk0 & 1u) && kk0 < kk1) {
        const float xv = x[kk0];
        if (row < out0_dim) sum0 += __half2float(wr0[kk0]) * xv;
        if (row < out1_dim) sum1 += __half2float(wr1[kk0]) * xv;
        kk0++;
    }
    const __half2 *wr02 = (const __half2 *)(wr0 + kk0);
    const __half2 *wr12 = (const __half2 *)(wr1 + kk0);
    const float2  *xr2  = (const float2  *)(x + kk0);
    const uint64_t n2 = (kk1 - kk0) / 2u;
    for (uint64_t i = 0; i < n2; i++) {
        float2 wf0 = __half22float2(wr02[i]);
        float2 wf1 = __half22float2(wr12[i]);
        float2 xf  = xr2[i];
        if (row < out0_dim) {
            sum0 += wf0.x * xf.x;
            sum0 += wf0.y * xf.y;
        }
        if (row < out1_dim) {
            sum1 += wf1.x * xf.x;
            sum1 += wf1.y * xf.y;
        }
    }
    if ((kk1 - kk0) & 1u) {
        uint64_t i = kk1 - 1u;
        const float xv = x[i];
        if (row < out0_dim) sum0 += __half2float(wr0[i]) * xv;
        if (row < out1_dim) sum1 += __half2float(wr1[i]) * xv;
    }
    partial0[tid] = sum0;
    partial1[tid] = sum1;
    __syncthreads();
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total0 += partial0[i];
            total1 += partial1[i];
        }
        if (row < out0_dim) out0[row] = total0;
        if (row < out1_dim) out1[row] = total1;
    }
}

__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    if (threadIdx.x == 0) out[tok * out_dim + row] = block_sum_f32(sum);
}

__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}
