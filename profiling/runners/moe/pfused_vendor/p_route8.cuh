// Vendored kernel source, temporary: delete once the kernel ships as a callable package.
// Origin: untracked development tree moe-copy-gap/lanes/P_fused/p_route8.cuh, tree HEAD d3563d86, file mtime 2026-11-12 17:42 PST.
// Only change: compile-time config is read from the module global _PFUSED_CFG (a dict) instead of os.environ.
// Lane P copy of lane O's o_route.cuh (do not edit O's file). Added: bits & 16 = zero out[T, H] bf16 rows of the
// block's own tokens at kernel start (moves the fused kernel's 8 MB zero-fill into the latency-bound router).
// bits & 64 = zero stores with an L2 evict_last policy (else __stcs).
// bits & 32 = set the router's preferred shared-memory carveout to 100% (host attribute, once per instantiation).
// Lane O: routing front end (counts, perm, route_w, swizzled SFB, permuted FP4 copy into [E, SLOTS]) for the routed
// NVFP4 MoE chain. Same outputs / slot order (ascending token) as lane N's v1 (n_route.cuh).
// Block b owns tokens [TB*b, TB*b+TB). x / x_sf rows, own ids and weights are loaded at kernel start (independent of
// routing). Ranks inside the block use __match_any_sync per warp + per-warp per-expert counts (no serial loop).
// MODE 0 ("all-ids"): prefix over earlier tokens = smem histogram of ids [0, t0) (v1 scheme; O(T^2) total reads).
// MODE 1 ("hist-barrier"): each block publishes its own per-expert histogram, one self-resetting grid barrier, then
//   each block sums the histograms of blocks < b (O(nb*E) per block). Needs all blocks co-resident (host checks).
// MODE 2/3/4 = floor kernels (timing only; slot taken from a precomputed table pslot[T*8]):
//   2 = copy floor (same loads/stores, no routing), 3 = 2 + load ids [0,t0) (design a load structure),
//   4 = 2 + publish hist + grid barrier + read hist of blocks < b (design b load structure).
// MODE 5 ("chunk-hist", 2 launches): o_hist_kernel writes per-64-token-chunk histograms ghist[c*64+e] (PDL trigger at
//   start); the main kernel (launched with programmatic stream serialization) issues its x/SF/id loads and the in-chunk
//   prefix (ids of [64*c, t0)) before griddepcontrol.wait, then sums ghist of chunks < c. O(T/64 * E) per block.
// MODE 6 ("fi-convert"): FlashInfer routing already ran; pslot = its expanded_idx_to_permuted_idx [T*8], ghist = its
//   num_tokens_per_expert [E], tile_n = bits >> 8. slot = permuted - padded expert offset (offsets = exclusive scan of
//   counts rounded up to tile_n, recomputed per block). Block 0 writes counts. Then the same SF swizzle + FP4 copy.
// lane P round 11 ledger bits: 4096 = skip the prefix id scan, 8192 = return after the id reads, 16384 = no perm/route_w
// stores, 32768 = no SFB (no x_sf loads, no swizzled stores).
// lane P round 12 P_FLAG (bits & 131072): flag area at counts + 8192 (r12b; ints): [0] gen (table ready, 1 = ready; the fused
//   kernel's last acking warp resets it), [1] fused ack counter, [2] watchdog errors, [3] perm-done block counter,
//   [16 + e] rows of expert e whose xg + SFB are written (monotone), [80 + e] target = cumulative min(count, SLOTS) (monotone).
//   Every block: perm/rw stores -> fence -> pdone++; after its copy + SFB: fence -> rdy[e] += its rows of e (release).
//   Last block: counts + table -> wait pdone == nb (all perm/rw visible) -> pdone = 0 -> target += counts -> gen = 1 (release).
// bits: 1 = PDL launch_dependents at start; 2 = stop after perm/route_w (through-perm); 4 = no FP4 copy.
__device__ __forceinline__ unsigned o_ld_acq(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}

template <int TB, int NT, int MODE>
__global__ void __launch_bounds__(NT, (MODE == 1 || MODE == 4) ? (1024 / NT) : 1) o_route_kernel(
    const int* __restrict__ ids, const float* __restrict__ tw, const int4* __restrict__ x4, const int* __restrict__ xsf,
    int* __restrict__ counts, int* __restrict__ perm, float* __restrict__ rw, int4* __restrict__ xg4, int* __restrict__ sfb,
    int* ghist, unsigned* gbar, const int* __restrict__ pslot, int bits, int E, int SLOTS, int4* __restrict__ zo) {
  constexpr int K = 8, SFW = 64, HW4 = 128, P = TB * K, NW = NT / 32, PW = (P + 31) / 32, U = 8;
  constexpr int XPT = (TB * HW4 + NT - 1) / NT, SPT = (TB * SFW + NT - 1) / NT, NS = NT / 16;
  static_assert(P <= NT && NT % 32 == 0 && NT >= 64, "shape");
  if (bits & 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, t0 = b * TB, nb = gridDim.x;
  if (bits & 16) {  // lane P: zero this block's out rows (H bf16 = 512 int4 per token), fire-and-forget
    int4* zr = zo + (size_t)t0 * 512;
#pragma unroll 4
    if (bits & 64) {  // keep the zeroed rows L2-resident (evict_last) until FC2's red.adds
      uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
      for (int i = tid; i < TB * 512; i += NT)
        asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1, %1, %1, %1}, %2;" ::"l"(zr + i), "r"(0), "l"(pol) : "memory");
    } else {
      for (int i = tid; i < TB * 512; i += NT) __stcs(zr + i, make_int4(0, 0, 0, 0));
    }
  }
  constexpr bool HP = (MODE == 0 || MODE == 5 || MODE == 7);  // smem-histogram prefix over ids
  constexpr int HR = (MODE == 7) ? 1 : NW;  // lane P round 4: MODE 7 = MODE 0 with one shared histogram row (less smem)
  __shared__ int s_h[HP ? HR : 1][64];
  __shared__ int s_wc[PW][64];
  __shared__ int s_id[P];
  __shared__ int s_slot[P];
  __shared__ int s_pre[64];
  __shared__ int s_tot[64];
  __shared__ int s_red[(MODE == 1 || MODE == 4 || MODE == 5) ? NS : 1][64];
  int4 xv[XPT];
  int sfv[SPT];
  const bool copy = !(bits & 4);
  if (copy) {
#pragma unroll
    for (int u = 0; u < XPT; ++u) { const int i = tid + u * NT; if (TB * HW4 % NT == 0 || i < TB * HW4) xv[u] = x4[(size_t)(t0 + i / HW4) * HW4 + i % HW4]; }
  }
#pragma unroll
  for (int u = 0; u < SPT; ++u) { const int i = tid + u * NT; if (!(bits & 32768) && (TB * SFW % NT == 0 || i < TB * SFW)) sfv[u] = xsf[(t0 + i / SFW) * SFW + i % SFW]; }
  int myid = -1, ps = -1; float myw = 0.f;
  if (tid < P) {
    myid = ids[t0 * K + tid]; myw = tw[t0 * K + tid];
    if (MODE >= 2 && MODE != 5 && MODE != 7) ps = pslot[t0 * K + tid];
  }
  for (int i = tid; i < PW * 64; i += NT) (&s_wc[0][0])[i] = 0;
  if (HP) for (int i = tid; i < HR * 64; i += NT) (&s_h[0][0])[i] = 0;
  __syncthreads();
  int xr = 0;
  if ((MODE == 0 || MODE == 3 || MODE == 5 || MODE == 7) && !(bits & 4096)) {  // ids of tokens [p0, t0) = int4 [2*p0, 2*t0)
    const int4* ids4 = (const int4*)ids;
    const int p0 = MODE == 5 ? (t0 & ~63) : 0;
    for (int c0 = 2 * p0; c0 < 2 * t0; c0 += NT * U) {
      int4 pv[U];
#pragma unroll
      for (int u = 0; u < U; ++u) { const int i = c0 + tid + u * NT; pv[u] = (i < 2 * t0) ? ids4[i] : make_int4(-1, -1, -1, -1); }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int4 a = pv[u];
        if (HP) {
          if ((unsigned)a.x < (unsigned)E) atomicAdd(&s_h[MODE == 7 ? 0 : wid][a.x], 1);
          if ((unsigned)a.y < (unsigned)E) atomicAdd(&s_h[MODE == 7 ? 0 : wid][a.y], 1);
          if ((unsigned)a.z < (unsigned)E) atomicAdd(&s_h[MODE == 7 ? 0 : wid][a.z], 1);
          if ((unsigned)a.w < (unsigned)E) atomicAdd(&s_h[MODE == 7 ? 0 : wid][a.w], 1);
        } else {
          xr ^= a.x ^ a.y ^ a.z ^ a.w;
        }
      }
    }
  }
  if (bits & 8192) {  // lane P round 11 ledger rung: stop after the id reads (sink keeps the loads alive)
    __syncthreads();
    if (tid < 64) ghist[b * 64 + tid] = (HP ? s_h[0][tid] : 0) ^ myid ^ xr ^ __float_as_int(myw);
    return;
  }
  // own pairs: rank among earlier pairs of the block with the same expert (top-k ids of one token are distinct)
  const int e = (tid < P && (unsigned)myid < (unsigned)E) ? myid : -1;
  int rk = 0;
  if (wid < PW) {
    const unsigned m = __match_any_sync(0xffffffffu, e);
    rk = __popc(m & ((1u << lane) - 1u));
    if (e >= 0 && rk == 0) s_wc[wid][e] = __popc(m);
  }
  if (tid < P) s_id[tid] = e;
  __syncthreads();
  if ((bits & 128) && tid < TB) {  // lane P round 4: per-token local-expert count (fused kernel's split zero)
    int c = 0;
#pragma unroll
    for (int k = 0; k < K; ++k) c += s_id[tid * K + k] >= 0;
    ((int*)zo)[t0 + tid] = c;
  }
  if (tid < 64) {
    int v = 0, o = 0;
    if (HP) {
#pragma unroll
      for (int w = 0; w < HR; ++w) v += s_h[w][tid];
    }
#pragma unroll
    for (int w = 0; w < PW; ++w) o += s_wc[w][tid];
    s_pre[tid] = v; s_tot[tid] = o;
    if (MODE == 1 || MODE == 4) ghist[b * 64 + tid] = o;
  }
  if (MODE == 1 || MODE == 4) {
    __syncthreads();
    if (tid == 0) {
      const unsigned gen = o_ld_acq(gbar + 1);
      __threadfence();
      const unsigned old = atomicAdd(gbar, 1u);
      if (old == (unsigned)nb - 1u) {
        atomicExch(gbar, 0u);
        __threadfence();
        atomicAdd(gbar + 1, 1u);
      } else {
        while (o_ld_acq(gbar + 1) == gen) {}
      }
      __threadfence();
    }
    __syncthreads();
    const int e4 = tid & 15, st = tid >> 4;
    int4 acc = make_int4(0, 0, 0, 0);
    const int4* gh4 = (const int4*)ghist;
#pragma unroll 8
    for (int j = st; j < b; j += NS) { const int4 v = __ldcg(gh4 + j * 16 + e4); acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
    s_red[st][e4 * 4 + 0] = acc.x; s_red[st][e4 * 4 + 1] = acc.y; s_red[st][e4 * 4 + 2] = acc.z; s_red[st][e4 * 4 + 3] = acc.w;
    __syncthreads();
    if (tid < 64) {
      int v = 0;
#pragma unroll
      for (int s = 0; s < NS; ++s) v += s_red[s][tid];
      s_pre[tid] = v;
    }
  }
  if (MODE == 6) {
    __syncthreads();
    if (tid == 0) {
      const int tn = bits >> 8;
      int off = 0;
      for (int q = 0; q < E; ++q) { s_pre[q] = off; const int c = ghist[q]; off += (c + tn - 1) / tn * tn; }
    }
    if (b == 0 && tid < E) counts[tid] = ghist[tid];
  }
  if (MODE == 5) {
    asm volatile("griddepcontrol.wait;" ::: "memory");  // chunk histograms complete
    const int e4 = tid & 15, st = tid >> 4, c = t0 >> 6;
    int4 acc = make_int4(0, 0, 0, 0);
    const int4* gh4 = (const int4*)ghist;
#pragma unroll 8
    for (int j = st; j < c; j += NS) { const int4 v = __ldcg(gh4 + j * 16 + e4); acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
    s_red[st][e4 * 4 + 0] = acc.x; s_red[st][e4 * 4 + 1] = acc.y; s_red[st][e4 * 4 + 2] = acc.z; s_red[st][e4 * 4 + 3] = acc.w;
    __syncthreads();
    if (tid < 64) {
      int v = s_pre[tid];
#pragma unroll
      for (int s = 0; s < NS; ++s) v += s_red[s][tid];
      s_pre[tid] = v;
    }
  }
  __syncthreads();
  if (tid < P) {
    int slot = -1;
    if (e >= 0) {
      if (MODE == 6) {
        slot = ps >= 0 ? ps - s_pre[e] : SLOTS;
      } else if (MODE >= 2 && MODE != 5 && MODE != 7) {
        slot = ps + (xr == 0x7fffffff ? 1 : 0);
      } else {
        int base = s_pre[e] + rk;
        for (int w = 0; w < wid; ++w) base += s_wc[w][e];
        slot = base;
      }
      if (slot < SLOTS) { if (!(bits & 16384)) { perm[e * SLOTS + slot] = t0 + tid / K; rw[e * SLOTS + slot] = myw; } } else slot = -1;
    }
    s_slot[tid] = slot;
  }
  if ((MODE <= 1 || MODE == 5 || MODE == 7) && b == nb - 1 && tid < E) counts[tid] = s_pre[tid] + s_tot[tid];
  int* fl = counts + 8192;  // r12b layout (ints): gen 0 | ack 32 | perm-done 64 | wd 96 | rdy[e] 128 + 32 e | target[e] 1280 + e
  __syncthreads();
  if ((bits & 131072) && tid == 0) { __threadfence(); atomicAdd(fl + 64, 1); }
  if ((bits & 768) && (MODE <= 1 || MODE == 5 || MODE == 7) && b == nb - 1) {
    // lane P round 8: the last block knows the totals -> write the fused kernel's tile list at counts + 64 (packed
    // m | n<<8 | e<<16 | phase<<24, FC1 then FC2, expert-major / token tile / weight tile fastest = the RR order;
    // ntc = 32 weight tiles per token tile, 16 pairs for P_CL=2), then 160 x -1 (end marker for every CTA stride).
    // round 8b: warp-0 shuffle scan + int4 stores spread over all threads (was tid0 serial scan + scalar per-tile stores)
    __shared__ int s_tp[65];
    const int ntc = (bits & 512) ? 16 : 32, q = ntc / 4;
    if (wid == 0) {
      const int e0 = lane * 2, e1 = e0 + 1;
      const int c0 = e0 < E ? (s_pre[e0] + s_tot[e0] + 127) / 128 : 0;
      const int c1 = e1 < E ? (s_pre[e1] + s_tot[e1] + 127) / 128 : 0;
      const int sm = c0 + c1; int x = sm;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
      if (e0 < E) s_tp[e0] = x - sm;
      if (e1 < E) s_tp[e1] = x - sm + c0;
      if (lane == 31) s_tp[E] = x;
    }
    __syncthreads();
    const int S = s_tp[E], half = S * ntc, sq = S * q;
    int* tab = counts + 64;
    int4* tab4 = (int4*)tab;
    for (int i = tid; i < 2 * sq; i += NT) {
      const int ph = i >= sq ? 1 : 0, j = i - ph * sq, u = j / q, m0 = (j % q) * 4;
      int lo = 0, hi = E - 1;
      while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (s_tp[mid] <= u) lo = mid; else hi = mid - 1; }
      const int v = ((u - s_tp[lo]) << 8) | (lo << 16) | (ph << 24) | m0;
      tab4[i] = make_int4(v, v + 1, v + 2, v + 3);
    }
    for (int i = tid; i < 160; i += NT) tab[2 * half + i] = -1;
    if (bits & 131072) {
      if (tid == 0) {
        long long n = 0;
        while (o_ld_acq((const unsigned*)(fl + 64)) < (unsigned)nb && ++n < (1ll << 24)) __nanosleep(64);
        if (n >= (1ll << 24)) atomicAdd(fl + 96, 1000000);
        fl[64] = 0;
      }
      if (tid < E) { const int c = s_pre[tid] + s_tot[tid]; fl[1280 + tid] += c < SLOTS ? c : SLOTS; }
      __syncthreads();
      if (tid == 0) { __threadfence(); asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(fl), "r"(1) : "memory"); }
    }
  }
  if (bits & 2) return;
  const int sfe = SLOTS / 128 * SFW * 128;  // SF ints per expert
#pragma unroll
  for (int u = 0; u < SPT; ++u) {
    if (bits & 32768) break;
    const int i = tid + u * NT, tt = i / SFW, kb4 = i % SFW;
    if (TB * SFW % NT != 0 && i >= TB * SFW) continue;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int ee = s_id[tt * K + k], s = s_slot[tt * K + k];
      if (ee >= 0 && s >= 0) sfb[ee * sfe + ((s / 128) * SFW + kb4) * 128 + (s % 32) * 4 + (s % 128) / 32] = sfv[u];
    }
  }
  if (copy) {
#pragma unroll
    for (int u = 0; u < XPT; ++u) {
      const int i = tid + u * NT, tt = i / HW4, col = i % HW4;
      if (TB * HW4 % NT != 0 && i >= TB * HW4) continue;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        const int ee = s_id[tt * K + k], s = s_slot[tt * K + k];
        if (ee >= 0 && s >= 0) xg4[((size_t)ee * SLOTS + s) * HW4 + col] = xv[u];
      }
    }
  }
  if (bits & 131072) {  // round 12: this block's xg rows + SFB of expert e are written -> rdy[e] += rows (release)
    __syncthreads();
    if (tid < E) {
      int n = 0;
      for (int j = 0; j < P; ++j) n += (s_id[j] == tid && s_slot[j] >= 0);
      if (n) { __threadfence(); asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(fl + 128 + 32 * tid), "r"(n) : "memory"); }
    }
  }
}

// per-64-token-chunk histogram of local expert ids (MODE 5 first launch). 128 threads, one int4 (4 ids) each.
__global__ void __launch_bounds__(128) o_hist_kernel(const int4* __restrict__ ids4, int* __restrict__ ghist, int E) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __shared__ int h[64];
  const int c = blockIdx.x, tid = threadIdx.x;
  if (tid < 64) h[tid] = 0;
  const int4 a = ids4[c * 128 + tid];
  __syncthreads();
  if ((unsigned)a.x < (unsigned)E) atomicAdd(&h[a.x], 1);
  if ((unsigned)a.y < (unsigned)E) atomicAdd(&h[a.y], 1);
  if ((unsigned)a.z < (unsigned)E) atomicAdd(&h[a.z], 1);
  if ((unsigned)a.w < (unsigned)E) atomicAdd(&h[a.w], 1);
  __syncthreads();
  if (tid < 64) ghist[c * 64 + tid] = h[tid];
}

__global__ void o_empty_kernel(int bits) {
  if (bits & 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

template <int TB, int NT, int MODE>
static void o_launch(int T, cudaStream_t st, const int* ids, const float* tw, const int4* x4, const int* xsf, int* counts,
                     int* perm, float* rw, int4* xg4, int* sfb, int* ghist, unsigned* gbar, const int* pslot, int bits,
                     int E, int slots, int4* zo) {
  const int nb = T / TB;
  if (MODE == 1 || MODE == 4) {
    int per_sm = 0, dev = 0, nsm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, o_route_kernel<TB, NT, MODE>, NT, 0);
    TORCH_CHECK(nb <= per_sm * nsm, "grid barrier needs co-residency: nb=", nb, " per_sm=", per_sm, " nsm=", nsm);
  }
  if (MODE == 5) {
    TORCH_CHECK(T % 64 == 0 && 64 % TB == 0);
    o_hist_kernel<<<T / 64, 128, 0, st>>>((const int4*)ids, ghist, E);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(nb); cfg.blockDim = dim3(NT); cfg.dynamicSmemBytes = 0; cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, o_route_kernel<TB, NT, MODE>, ids, tw, x4, xsf, counts, perm, rw, xg4, sfb, ghist,
                                   gbar, pslot, bits, E, slots, zo) == cudaSuccess);
    return;
  }
  if (bits & 32) {  // lane P: max-shared carveout so a following large-smem kernel can co-reside / launch without an SM reconfig
    static bool done = false;
    if (!done) { cudaFuncSetAttribute(o_route_kernel<TB, NT, MODE>, cudaFuncAttributePreferredSharedMemoryCarveout, 100); done = true; }
  }
  o_route_kernel<TB, NT, MODE><<<nb, NT, 0, st>>>(ids, tw, x4, xsf, counts, perm, rw, xg4, sfb, ghist, gbar, pslot, bits, E, slots, zo);
}

void o_route(torch::Tensor ids, torch::Tensor tw, torch::Tensor x, torch::Tensor xsf, torch::Tensor counts,
             torch::Tensor perm, torch::Tensor rw, torch::Tensor xg, torch::Tensor sfb, torch::Tensor ghist,
             torch::Tensor gbar, torch::Tensor pslot, int64_t E, int64_t slots, int64_t tb, int64_t nt, int64_t mode,
             int64_t bits, torch::Tensor zout) {
  auto st = c10::cuda::getCurrentCUDAStream();
  const int T = (int)ids.size(0);
  TORCH_CHECK(T % tb == 0 && slots % 128 == 0 && ids.size(1) == 8 && x.size(1) == 2048 && E <= 64);
  TORCH_CHECK(mode == 6 || mode == 0 || mode == 2 || mode == 3 || ghist.numel() >= (T / tb) * 64);
#define O_ARGS T, st, (const int*)ids.data_ptr(), (const float*)tw.data_ptr(), (const int4*)x.data_ptr(), \
    (const int*)xsf.data_ptr(), (int*)counts.data_ptr(), (int*)perm.data_ptr(), (float*)rw.data_ptr(), \
    (int4*)xg.data_ptr(), (int*)sfb.data_ptr(), (int*)ghist.data_ptr(), (unsigned*)gbar.data_ptr(), \
    (const int*)pslot.data_ptr(), (int)bits, (int)E, (int)slots, (int4*)zout.data_ptr()
#define O_CASE(TB_, NT_)                                         \
  if (tb == TB_ && nt == NT_) {                                  \
    switch (mode) {                                              \
      case 0: o_launch<TB_, NT_, 0>(O_ARGS); break;              \
      case 1: o_launch<TB_, NT_, 1>(O_ARGS); break;              \
      case 2: o_launch<TB_, NT_, 2>(O_ARGS); break;              \
      case 3: o_launch<TB_, NT_, 3>(O_ARGS); break;              \
      case 4: o_launch<TB_, NT_, 4>(O_ARGS); break;              \
      case 5: o_launch<TB_, NT_, 5>(O_ARGS); break;              \
      case 6: o_launch<TB_, NT_, 6>(O_ARGS); break;              \
      case 7: o_launch<TB_, NT_, 7>(O_ARGS); break;              \
      default: TORCH_CHECK(false, "mode");                       \
    }                                                            \
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);              \
    return;                                                      \
  }
  O_CASE(1, 64) O_CASE(1, 128) O_CASE(2, 128) O_CASE(2, 256) O_CASE(4, 128) O_CASE(4, 256) O_CASE(8, 128)
  O_CASE(8, 256) O_CASE(16, 256) O_CASE(16, 512) O_CASE(32, 256) O_CASE(32, 512)
  TORCH_CHECK(false, "no instantiation for tb/nt ", tb, "/", nt);
}

// lane P round 4: static smem / regs of one instantiation (for the co-residency budget)
std::vector<int64_t> o_attr(int64_t tb, int64_t nt, int64_t mode) {
  cudaFuncAttributes a{};
#define O_ATTR(TB_, NT_, M_) if (tb == TB_ && nt == NT_ && mode == M_) cudaFuncGetAttributes(&a, o_route_kernel<TB_, NT_, M_>);
  O_ATTR(2, 128, 0) O_ATTR(2, 128, 7) O_ATTR(2, 128, 5) O_ATTR(4, 128, 0) O_ATTR(4, 128, 7) O_ATTR(8, 256, 0) O_ATTR(8, 256, 7)
  O_ATTR(16, 256, 0) O_ATTR(16, 256, 7) O_ATTR(8, 128, 0) O_ATTR(8, 128, 7) O_ATTR(16, 256, 5) O_ATTR(8, 256, 5)
  return {(int64_t)a.sharedSizeBytes, (int64_t)a.numRegs, (int64_t)a.maxThreadsPerBlock};
}

void o_empty(int64_t nblocks, int64_t bits) {
  auto st = c10::cuda::getCurrentCUDAStream();
  o_empty_kernel<<<(int)nblocks, 32, 0, st>>>((int)bits);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

// lane P round 10: self-route FLOOR (timing only). Every block reads all T*8 ids (int4, contiguous range per thread),
// builds the local-expert histogram (smem atomics), the tile prefix (warp-0 shuffle scan), and with bits & 2 the
// ordered row list of one expert (= the expert of its first tile) by count -> block exclusive scan -> scatter.
// bits & 1: launch_dependents at start; bits & 4: griddepcontrol.wait before reading ids. Writes 2 ints per block.
__global__ void __launch_bounds__(256) o_sfloor_kernel(const int4* __restrict__ ids4, int* __restrict__ out, int T, int E, int bits) {
  if (bits & 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (bits & 4) asm volatile("griddepcontrol.wait;" ::: "memory");
  __shared__ int s_h[64];
  __shared__ int s_tp[65];
  __shared__ int s_w[32];
  __shared__ int s_rows[4096];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, NT = blockDim.x, NW = NT >> 5;
  if (tid < 64) s_h[tid] = 0;
  __syncthreads();
  const int n4 = T * 2, R = (n4 + NT - 1) / NT, i0 = tid * R, i1 = min(i0 + R, n4);
  const int e0 = (blockIdx.x / 32) % E;
  int cnt = 0;
  for (int j = i0; j < i1; j += 8) {
    int4 v[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) v[u] = (j + u < i1) ? __ldg(ids4 + j + u) : make_int4(-1, -1, -1, -1);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int4 a = v[u];
      if ((unsigned)a.x < (unsigned)E) atomicAdd(&s_h[a.x], 1);
      if ((unsigned)a.y < (unsigned)E) atomicAdd(&s_h[a.y], 1);
      if ((unsigned)a.z < (unsigned)E) atomicAdd(&s_h[a.z], 1);
      if ((unsigned)a.w < (unsigned)E) atomicAdd(&s_h[a.w], 1);
      cnt += (a.x == e0) + (a.y == e0) + (a.z == e0) + (a.w == e0);
    }
  }
  __syncthreads();
  if (wid == 0) {
    const int ea = lane * 2, eb = ea + 1;
    const int c0 = ea < E ? (s_h[ea] + 127) / 128 : 0, c1 = eb < E ? (s_h[eb] + 127) / 128 : 0;
    const int sm = c0 + c1; int x = sm;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
    if (ea < E) s_tp[ea] = x - sm;
    if (eb < E) s_tp[eb] = x - sm + c0;
    if (lane == 31) s_tp[E] = x;
  }
  int nrows = 0;
  if (bits & 2) {  // ordered row list of expert e0 (token order = entry order; a token's top-k ids are distinct)
    int x = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
    if (lane == 31) s_w[wid] = x;
    __syncthreads();
    if (wid == 0) {
      int w = lane < NW ? s_w[lane] : 0;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, w, o); if (lane >= o) w += y; }
      if (lane < NW) s_w[lane] = w;
    }
    __syncthreads();
    int pos = x - cnt + (wid > 0 ? s_w[wid - 1] : 0);
    for (int j = i0; j < i1; ++j) {
      const int4 a = __ldg(ids4 + j);
      if (a.x == e0 || a.y == e0 || a.z == e0 || a.w == e0) { if (pos < 4096) s_rows[pos] = j >> 1; ++pos; }
    }
    __syncthreads();
    nrows = s_w[NW - 1];
  } else {
    __syncthreads();
  }
  if (tid == 0) { out[blockIdx.x * 2] = s_tp[E]; out[blockIdx.x * 2 + 1] = nrows > 0 ? s_rows[nrows - 1] : -1; }
}

void o_sfloor(torch::Tensor ids, torch::Tensor out, int64_t E, int64_t nblocks, int64_t nt, int64_t bits) {
  auto st = c10::cuda::getCurrentCUDAStream();
  const int T = (int)ids.size(0);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((int)nblocks); cfg.blockDim = dim3((int)nt); cfg.dynamicSmemBytes = 0; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = (bits & 4) ? 1 : 0;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, o_sfloor_kernel, (const int4*)ids.data_ptr(), (int*)out.data_ptr(), T, (int)E, (int)bits) == cudaSuccess);
}
