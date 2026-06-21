#include "ds4_gpu.h"

int ds4_gpu_dsv4_compressed_kv_quantize_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot)
{ (void)x; (void)n_tok; (void)head_dim; (void)n_rot; return 1; }
bool ds4_gpu_dsv4_turbo4_packv4_enabled(void) { return false; }
uint64_t ds4_gpu_dsv4_turbo4_packed_kv_row_bytes(uint32_t hd, uint32_t nr) { (void)hd; (void)nr; return 0; }
uint64_t ds4_gpu_dsv4_turbo4_packed_kv_bytes(uint32_t nr, uint32_t hd, uint32_t nrot) { (void)nr; (void)hd; (void)nrot; return 0; }
int ds4_gpu_dsv4_turbo4_pack_compressed_kv_tensor(
        ds4_gpu_tensor *p, const ds4_gpu_tensor *s, uint32_t dr, uint32_t sr,
        uint32_t nr, uint32_t hd, uint32_t nrot) { (void)p; (void)s; (void)dr; (void)sr; (void)nr; (void)hd; (void)nrot; return 0; }
int ds4_gpu_dsv4_turbo4_unpack_compressed_kv_tensor(
        ds4_gpu_tensor *d, const ds4_gpu_tensor *p, uint32_t dr, uint32_t sr,
        uint32_t nr, uint32_t hd, uint32_t nrot) { (void)d; (void)p; (void)dr; (void)sr; (void)nr; (void)hd; (void)nrot; return 0; }
int ds4_gpu_attention_indexed_mixed_batch_heads_turbo4_tensor(
        ds4_gpu_tensor *heads, const void *mm, uint64_t ms, uint64_t soff,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *rkv, const ds4_gpu_tensor *ckv,
        const ds4_gpu_tensor *tk, uint32_t nt, uint32_t p0, uint32_t nr,
        uint32_t rc, uint32_t rs, uint32_t nc, uint32_t t_k, uint32_t w,
        uint32_t ratio, uint32_t nh, uint32_t hd) {
    (void)heads; (void)mm; (void)ms; (void)soff; (void)q; (void)rkv; (void)ckv;
    (void)tk; (void)nt; (void)p0; (void)nr; (void)rc; (void)rs; (void)nc;
    (void)t_k; (void)w; (void)ratio; (void)nh; (void)hd; return 0; }
