# VENDORED (temporary, delete once the kernel ships as a callable package): wt-glm53-decode-investigate/tmp/moe-copy-gap/lanes/P_fused/p_route.py
# MoE-session lane P, round R16; source tree HEAD d3563d86 (file untracked), mtime 2026-11-12 17:42:53 PST.
# Only change: compile-time config is read from the module global _PFUSED_CFG (a dict) instead of os.environ.
"""Lane P copy of lane O's router (p_route.cuh, + zero-fill of out in the router, bits & 16). ORouter has the same output attributes as lane N's RouterN."""
import os
import torch
from torch.utils.cpp_extension import load_inline

HDR = "#include <torch/extension.h>\n#include <cuda_runtime.h>\n#include <c10/cuda/CUDAStream.h>\n"
CPP = ("void o_route(torch::Tensor ids, torch::Tensor tw, torch::Tensor x, torch::Tensor xsf, torch::Tensor counts, "
       "torch::Tensor perm, torch::Tensor rw, torch::Tensor xg, torch::Tensor sfb, torch::Tensor ghist, torch::Tensor gbar, "
       "torch::Tensor pslot, int64_t E, int64_t slots, int64_t tb, int64_t nt, int64_t mode, int64_t bits, torch::Tensor zout);"
       "void o_empty(int64_t nblocks, int64_t bits); void o_sfloor(torch::Tensor ids, torch::Tensor out, int64_t E, int64_t nblocks, int64_t nt, int64_t bits); std::vector<int64_t> o_attr(int64_t tb, int64_t nt, int64_t mode);")
_mods = {}


def mod(build_dir):
    """Build (once per build_dir) the router extension from the vendored p_route8.cuh (was P_ROUTE_SRC / P_EXT_DIR env)."""
    if build_dir not in _mods:
        here = os.path.dirname(os.path.abspath(__file__))
        src = HDR + open(os.path.join(here, "p_route8.cuh")).read()
        os.makedirs(build_dir, exist_ok=True)
        _mods[build_dir] = load_inline("p_route_ext", CPP, cuda_sources=src, functions=["o_route", "o_empty", "o_attr", "o_sfloor"],
                                       build_directory=build_dir, extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"], verbose=False)
    return _mods[build_dir]


class PRouter:
    """cfg = (TB, NT, MODE). MODE 0 all-ids prefix, 1 hist + grid barrier, 2/3/4 floors (need pslot)."""

    def __init__(self, T, K, E, H, slots, cfg, copy=True, pdl=True, through_perm=False, shared=None, zout=None, carve=False, zlast=False, nloc=None, build_dir=None):
        assert K == 8 and H == 4096 and slots % 128 == 0
        self.tb, self.nt, self.mode = cfg
        assert T % self.tb == 0
        self.E, self.slots = E, slots
        self.bits = int(pdl) | (2 if through_perm else 0) | (0 if copy else 4) | (16 if zout is not None else 0) | (32 if carve else 0) | (64 if zlast else 0) | (128 if nloc is not None else 0)
        self.zout = zout if zout is not None else torch.zeros(16, dtype=torch.int32, device="cuda")
        if nloc is not None:
            assert zout is None
            self.zout = nloc
        dev = "cuda"
        if shared is not None:  # share output buffers with another router (e.g. floors writing into the same tensors)
            self.counts, self.perm, self.route_w, self.sfb, self.xg = shared.counts, shared.perm, shared.route_w, shared.sfb, shared.xg
        else:
            self.cbuf = torch.full((64 + 16384,), -1, dtype=torch.int32, device=dev)  # counts + tile table (round 8)
            self.cbuf[:64].zero_()
            self.cbuf[8192:8192 + 1344].zero_()  # round 12 P_FLAG flag area (r12b layout)
            self.counts = self.cbuf[:E]
            self.perm = torch.full((E, slots), -1, dtype=torch.int32, device=dev)
            self.route_w = torch.zeros(E, slots, dtype=torch.float32, device=dev)
            self.sfb = torch.zeros(E, slots // 128, H // 64, 32, 4, 4, dtype=torch.uint8, device=dev)
            self.xg = torch.zeros(E, slots, H // 2, dtype=torch.uint8, device=dev)
        self.ghist = torch.zeros(T // self.tb * 64, dtype=torch.int32, device=dev)
        self.gbar = torch.zeros(2, dtype=torch.int32, device=dev)
        self.pslot = torch.full((T * 8,), -1, dtype=torch.int32, device=dev)
        self.m = mod(build_dir)

    def set_pslot(self, ids):
        """Precomputed slot table (host) for the floor modes: slot of (t, k) in expert ids[t, k], -1 if not local."""
        ids = ids.cpu().long()
        T = ids.shape[0]
        ps = torch.full((T, 8), -1, dtype=torch.int32)
        c = [0] * self.E
        for t in range(T):
            for k in range(8):
                e = int(ids[t, k])
                if 0 <= e < self.E:
                    ps[t, k] = c[e]; c[e] += 1
        self.pslot.copy_(ps.flatten().cuda())

    def __call__(self, ids, tw, x, xsf):
        self.m.o_route(ids, tw, x, xsf, self.counts, self.perm, self.route_w, self.xg, self.sfb, self.ghist, self.gbar,
                       self.pslot, self.E, self.slots, self.tb, self.nt, self.mode, self.bits, self.zout)



