# VENDORED (temporary, delete once the kernel ships as a callable package): wt-glm53-decode-investigate/tmp/moe-copy-gap/lanes/P_fused/p_fused.py
# MoE-session lane P, round R16; source tree HEAD d3563d86 (file untracked), mtime 2026-11-12 16:02:19 PST.
# Only change: compile-time config is read from the module global _PFUSED_CFG (a dict) instead of os.environ.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
import functools
import os
_CFG = globals().get('_PFUSED_CFG') or {}
_GK=int(_CFG.get('G_KTILE','4')); _GC=int(_CFG.get('G_CSTAGE','2')); _GCR=int(_CFG.get('G_CREFINE','1')); _GAB=int(_CFG.get('G_ABSTAGE','0'))
_HF=int(_CFG.get('H_FUSED','1')); _HS=int(_CFG.get('H_SCHED','0')); _HEPI_COLS=int(_CFG.get('P_EC','32')); _HNOEPI=int(_CFG.get('H_NOEPI','0')); _HACC=int(_CFG.get('H_ACC','0')); _HBST=int(_CFG.get('H_BSTAGE','4')); _HZB=int(_CFG.get('H_ZBYTES','0')); _HZPOS=int(_CFG.get('H_ZPOS','0')); _HPDLT=int(_CFG.get('H_PDLTRIG','0')); _HPDLW=int(_CFG.get('H_PDLWAIT','0')); _HPF=int(_CFG.get('H_PREPF','0'))
_JX=int(_CFG.get('J_X','0')); _JV=int(_CFG.get('J_VN','0')); _JD=int(_CFG.get('J_DYN','0')); _JQ=int(_CFG.get('J_Q','4')); _JS=int(_CFG.get('J_SPLIT','0')); _JW1=4096; _JSMAX=192; _JT=int(_CFG.get('J_TRACE','0')); _JMAX=8192 + 160 * 64
_PF=1; _PG=int(_CFG.get('P_GATE','1')); _PZ=int(_CFG.get('P_ZERO','1')); _PNG=36*64; _PSTOP=int(_CFG.get('P_STOP','0')); _PNOEPI=int(_CFG.get('P_NOEPI','0')); _PNOB=int(_CFG.get('P_NOB','0')); _PSLEEP=int(_CFG.get('P_SLEEP','256')); _PRLX=int(_CFG.get('P_RLX','0')); _PTR=int(_CFG.get('P_TRACE','0')); _PGL=int(_CFG.get('P_GL','1')); _PUNI=int(_CFG.get('P_UNI','1')); _PPFN=int(_CFG.get('P_PFN','1')); _PEVF=int(_CFG.get('P_EVF','0')); _PSPL=int(_CFG.get('P_SPL','0')); _PEA=int(_CFG.get('P_EARLY','0')); _PEAX=5; _PZT=int(_CFG.get('P_ZT','0')); _PNL0=2*36*64+8+8*160; _PK=int(_CFG.get('P_K2','1')); _PCL=int(_CFG.get('P_CL','1')); _PASK=int(_CFG.get('P_ASK','0')); _PNORED=int(_CFG.get('P_NORED','0')); _PORD=int(_CFG.get('P_ORD','0')); _PZTL=int(_CFG.get('P_ZTAIL','0')); _PNOLD=int(_CFG.get('P_NOLD','0')); _PTDB=int(_CFG.get('P_TDB','0')); _PPFL2=int(_CFG.get('P_PFL2','32'))
_GNOMMA=int(_CFG.get('G_NOMMA','0')); _GNOS2T=int(_CFG.get('G_NOS2T','0')); _GNOB=int(_CFG.get('G_NOB','0')); _GNOC=int(_CFG.get('G_NOC','0'))
from cutlass._mlir import ir
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

from cutlass.cutlass_dsl import (
    Int32,
    Int64,
    Uint8,
    Uint64,
    T,
    Integer,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass._mlir.dialects import llvm
from flashinfer.utils import get_compute_capability
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    cutlass_to_torch_dtype,
    get_num_sm,
    get_max_active_clusters,
    make_ptr,
)
from typing import Callable, List


sizeof_i32 = 4


@dsl_user_op
def with_byte(obj: Uint64, index: Int32, value: Uint8, *, loc=None, ip=None) -> Uint64:
    obj &= ~(0xFF << (index * 8))
    obj |= value << (index * 8)
    assert isinstance(obj, Uint64), f"{obj=}"
    return obj


@dsl_user_op
def read_byte(obj: Uint64, index: Int32, *, loc=None, ip=None) -> Uint8:
    return ((obj >> (index * 8)) & 0xFF).to(Uint8)


@dsl_user_op
def atomic_add_release_global(addr: Int64, value: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                addr.ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.add.release.gpu.global.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def j_ld_acquire(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [addr.ir_value(loc=loc, ip=ip)], "ld.acquire.gpu.global.s32 $0, [$1];", "=r,l",
                                 has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def j_fence_gpu(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "fence.acq_rel.gpu;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def j_gtimer(*, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [], "mov.u32 $0, %globaltimer_lo;", "=r",
                                 has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def h_cvt_e2m1x8(v0, v1, v2, v3, v4, v5, v6, v7, *, loc=None, ip=None) -> cutlass.Uint32:
    """8 fp32 -> 8 e2m1 nibbles packed in u32 (v0 in the low nibble), RN satfinite."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in (v0, v1, v2, v3, v4, v5, v6, v7)],
            "{ .reg .b8 b0, b1, b2, b3;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;\n"
            "mov.b32 $0, {b0, b1, b2, b3}; }",
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def h_e4m3_code(v, *, loc=None, ip=None) -> cutlass.Uint32:
    """fp32 -> e4m3 byte code (RN satfinite) in the low byte of a u32."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(v).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, $1, $1; cvt.u32.u16 $0, t; }",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    ) & cutlass.Uint32(0xFF)


@dsl_user_op
def h_e4m3_decode(code, *, loc=None, ip=None) -> cutlass.Float32:
    """e4m3 byte code (low byte of u32) -> fp32."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Uint32(code).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 t, lo, hi; .reg .b32 h2; cvt.u16.u32 t, $1; cvt.rn.f16x2.e4m3x2 h2, t; mov.b32 {lo, hi}, h2; cvt.f32.f16 $0, lo; }",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class DynScheduler:
    """Lane J dynamic list scheduler. Host provides order[0] = n_items, order[1..] = items
    packed (l << 16) | (n << 8) | m in the desired (hot-first) order. The TMA-producer warp of each CTA
    claims the next item with a global atomicAdd on ctr[0] and publishes it through a small smem ring
    (j_buf + full/empty mbarriers) to the MMA warp and the 4 epilogue warps (160 consumer arrivals).
    Self-resetting counter: every CTA performs exactly one failing claim; the claim that returns
    n_items + gridsize - 1 is the globally last one and stores ctr[0] = 0 (graph-replay safe)."""

    def __init__(self, ctx, is_producer, seq, num_tiles_executed, expand=None, cur=None, n=None):
        # expand = (masked_m, tile_n): epilogue of weight-once units walks the unit's token subtiles itself
        self.ctx = ctx
        self.is_producer = is_producer
        self._seq = seq
        self._num_tiles_executed = num_tiles_executed
        self.expand = expand
        self._cur = Int32(-1) if cur is None else cur
        self._n = Int32(0) if n is None else n

    def __extract_mlir_values__(self):
        values = extract_mlir_values(self._seq)
        values.extend(extract_mlir_values(self._num_tiles_executed))
        values.extend(extract_mlir_values(self._cur))
        values.extend(extract_mlir_values(self._n))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 4
        return DynScheduler(self.ctx, self.is_producer, new_from_mlir_values(self._seq, [values[0]]),
                            new_from_mlir_values(self._num_tiles_executed, [values[1]]), self.expand,
                            new_from_mlir_values(self._cur, [values[2]]), new_from_mlir_values(self._n, [values[3]]))

    @cute.jit
    def _get(self):
        if cutlass.const_expr(self.expand is not None):
            masked_m, tile_n = self.expand
            cur = self._cur
            n = self._n
            seq = self._seq
            more = cutlass.Boolean(False)
            if cur >= 0:
                if (n + 1) * tile_n < masked_m[cur >> 16]:
                    more = cutlass.Boolean(True)
            if more:
                n = n + 1
            else:
                cur = self._fetch(seq)
                seq = seq + 1
                n = Int32(0)
            self._cur = cur
            self._n = n
            self._seq = seq
            coord = (Int32(cur & 0xFF), Int32(n), Int32(cur >> 16))
            return WorkTileInfo(coord, cur >= 0)
        elif cutlass.const_expr(_JS):
            idx = self._fetch(self._seq)
            order = self.ctx[0]
            w0 = Int32(-1)
            w1 = Int32(0)
            if idx >= 0:
                w0 = order[idx + 1]
                w1 = order[_JW1 + idx]
            coord = (Int32(w0 & 0xFF), Int32((w0 >> 8) & 0xFF), Int32(w0 >> 16), w1)
            return WorkTileInfo(coord, idx >= 0)
        else:
            item = self._fetch(self._seq)
            coord = (Int32(item & 0xFF), Int32((item >> 8) & 0xFF), Int32(item >> 16))
            return WorkTileInfo(coord, item >= 0)

    def _fetch(self, seq):
        if cutlass.const_expr(_JD == 2):
            return self._fetch_static(seq)
        else:
            return self._fetch_dyn(seq)

    @cute.jit
    def _fetch_static(self, seq):
        # static per-CTA lists (host LPT): order[1 + cta * 24 + seq], -1 terminated; no claim/broadcast
        order = self.ctx[0]
        bx, by, bz = cute.arch.block_idx()
        it = Int32(-1)
        if seq < 24:
            it = order[1 + bz * 24 + seq]
        return it

    @cute.jit
    def _fetch_dyn(self, seq):
        order, ctr, buf, full, empty = self.ctx
        slot = seq % _JQ
        ph = (seq // _JQ) % 2
        if cutlass.const_expr(self.is_producer):
            cute.arch.mbarrier_wait(empty + slot, ph ^ 1)
            if cute.arch.lane_idx() == 0:
                idx = atomic_add_release_global(ctr.iterator.toint(), Int32(1))
                total = order[0]
                item = Int32(-1)
                if idx < total:
                    if cutlass.const_expr(_JS):
                        item = idx
                    else:
                        item = order[idx + 1]
                else:
                    gx, gy, gz = cute.arch.grid_dim()
                    if idx == total + gx * gy * gz - 1:
                        ctr[0] = Int32(0)
                buf[slot] = item
                cute.arch.mbarrier_arrive(full + slot)
            cute.arch.sync_warp()
        cute.arch.mbarrier_wait(full + slot, ph)
        item = buf[slot]
        if cutlass.const_expr(not self.is_producer):
            cute.arch.mbarrier_arrive(empty + slot)
        return item

    @dsl_user_op
    def get_current_work(self, dsm_pending_packed=None, dsm_counter=None, num_c_stage=None, *, loc=None, ip=None):
        return self._get(), dsm_pending_packed

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self._get()

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        if cutlass.const_expr(self.expand is None):
            self._seq += Int32(1)
        self._num_tiles_executed += Int32(1)

    @dsl_user_op
    def advance_to_next_unit(self, *, loc=None, ip=None):
        self._seq += Int32(1)
        self._num_tiles_executed += Int32(1)

    @property
    def num_tiles_executed(self):
        return self._num_tiles_executed


class UnitScheduler:
    """Weight-major persistent schedule (lane H step b): a work unit is (weight m-tile, expert);
    the CTA that owns a unit runs all of that expert's token tiles back to back (n fastest),
    so no two CTAs ever stream the same weight tile. Unit order: expert slow, m-tile fast.
    Same interface as MaskedScheduler (tile coord = (m, n, l))."""

    def __init__(self, params, num_persistent_clusters, unit, n, num_tiles_executed):
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self._unit = unit
        self._n = n
        self._num_tiles_executed = num_tiles_executed

    def __extract_mlir_values__(self):
        values = extract_mlir_values(self.num_persistent_clusters)
        values.extend(extract_mlir_values(self._unit))
        values.extend(extract_mlir_values(self._n))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 4
        return UnitScheduler(
            self.params,
            new_from_mlir_values(self.num_persistent_clusters, [values[0]]),
            new_from_mlir_values(self._unit, [values[1]]),
            new_from_mlir_values(self._n, [values[2]]),
            new_from_mlir_values(self._num_tiles_executed, [values[3]]),
        )

    @dsl_user_op
    @staticmethod
    def create(params, block_idx, grid_dim, *, loc=None, ip=None):
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip)
        bidx, bidy, bidz = block_idx
        return UnitScheduler(params, Int32(num_persistent_clusters), Int32(bidz), Int32(0), Int32(0))

    @cute.jit
    def _get(self):
        num_m = self.params.problem_shape_ntile_mnl[0]
        tile_n = self.params.c_tiler[1]
        num_l = self.params.masked_m.shape[0]
        total = num_m * num_l
        u = self._unit
        n = self._n
        keep = u < total
        while keep:
            cnt = self.params.masked_m[u // num_m]
            if cnt > n * tile_n:
                keep = cutlass.Boolean(False)
            else:
                u += self.num_persistent_clusters
                n = Int32(0)
                keep = u < total
        self._unit = u
        self._n = n
        is_valid = u < total
        coord = (Int32(u % num_m), Int32(n), Int32(u // num_m))
        return WorkTileInfo(coord, is_valid)

    @dsl_user_op
    def get_current_work(self, dsm_pending_packed=None, dsm_counter=None, num_c_stage=None, *, loc=None, ip=None):
        return self._get(), dsm_pending_packed

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self._get()

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        self._n += Int32(1)
        self._num_tiles_executed += Int32(1)

    @dsl_user_op
    def advance_to_next_unit(self, *, loc=None, ip=None):
        self._n = Int32(1 << 16)

    @property
    def num_tiles_executed(self):
        return self._num_tiles_executed


class FusedScheduler:
    """Lane P: static round robin over one list = all FC1 tiles in masked order (expert 0..L-1, token tile, weight
    tile fastest) followed by all FC2 tiles in the same order. tile coord = (m, n, l, phase)."""

    def __init__(self, params, nclu, lin, batch, accum, phase, base, nexec):
        self.params = params
        self.num_persistent_clusters = nclu
        self._lin = lin
        self._batch = batch
        self._accum = accum
        self._phase = phase
        self._base = base
        self._num_tiles_executed = nexec

    def __extract_mlir_values__(self):
        values = []
        for v in (self.num_persistent_clusters, self._lin, self._batch, self._accum, self._phase, self._base,
                  self._num_tiles_executed):
            values.extend(extract_mlir_values(v))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 7
        olds = (self.num_persistent_clusters, self._lin, self._batch, self._accum, self._phase, self._base,
                self._num_tiles_executed)
        news = [new_from_mlir_values(o, [v]) for o, v in zip(olds, values)]
        return FusedScheduler(self.params, *news)

    @dsl_user_op
    @staticmethod
    def create(params, block_idx, grid_dim, *, loc=None, ip=None):
        nclu = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(params.cluster_shape_mn, loc=loc, ip=ip)
        bidx, bidy, bidz = block_idx
        return FusedScheduler(params, nclu, Int32(bidz), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0))

    @cute.jit
    def _get_ord(self):
        # round 7: weight-tile-major order within each phase: for m: for expert: for token tile. Every round of G tiles
        # mixes all experts (first-touch DRAM tiles and L2-hit repeat token tiles at the global ratio); the k token
        # tiles of one (expert, m) are consecutive -> run concurrently on neighbouring CTAs (A read once from DRAM).
        NT = self.params.problem_shape_ntile_mnl[0]
        if cutlass.const_expr(_PCL > 1):
            NT = NT // _PCL
        num_batches = self.params.masked_m.shape[0]
        S = Int32(0)
        for b in cutlass.range(num_batches, unroll=1):
            S += (self.params.masked_m[b] + (self.params.c_tiler[1] - 1)) // self.params.c_tiler[1]
        tot = S * NT
        phase = self._phase
        base = self._base
        idx = self._lin - base
        if phase == 0 and idx >= tot and cutlass.const_expr(_PSTOP == 0):
            phase = Int32(1)
            base = base + tot
            idx = idx - tot
        self._phase = phase
        self._base = base
        Sx = max(S, Int32(1))
        mi = idx // Sx
        r = idx - mi * Sx
        batch = Int32(0)
        accum = Int32(0)
        keep = cutlass.Boolean(True)
        while keep:
            if batch < num_batches:
                nt = (self.params.masked_m[batch] + (self.params.c_tiler[1] - 1)) // self.params.c_tiler[1]
                if accum + nt <= r:
                    accum += nt
                    batch += Int32(1)
                else:
                    keep = cutlass.Boolean(False)
            else:
                keep = cutlass.Boolean(False)
        if idx >= tot:
            batch = Int32(num_batches)
        self._batch = batch
        cm = mi
        if cutlass.const_expr(_PCL > 1):
            cm = cm * _PCL + cute.arch.block_idx()[0] % _PCL
        coord = (cm, r - accum, batch, phase & 1)
        return WorkTileInfo(coord, batch < num_batches)

    @cute.jit
    def _get_rr(self):
        NT = self.params.problem_shape_ntile_mnl[0]  # weight tiles per token tile (swap-AB: 4096/128 = 32)
        if cutlass.const_expr(_PCL > 1):
            NT = NT // _PCL  # round 6: a cluster of _PCL CTAs (along M) takes _PCL adjacent weight tiles of one token tile
        num_batches = self.params.masked_m.shape[0]
        batch = self._batch
        accum = self._accum
        phase = self._phase
        base = self._base
        idx = self._lin - base
        hf = Int32(0)
        if cutlass.const_expr(_PSPL > 0):
            if phase == 1:
                idx, hf = self._pmap(base)
        keep = cutlass.Boolean(True)
        while keep:
            lim = num_batches
            if cutlass.const_expr(_PEA > 0):
                if phase == 0:
                    lim = min(num_batches, _PEAX)
            if batch < lim:
                nt = (self.params.masked_m[batch] + (self.params.c_tiler[1] - 1)) // self.params.c_tiler[1]
                if cutlass.const_expr(_PEA > 0):
                    if phase == 0:
                        nt = min(nt, 1)
                    elif phase == 2 and batch < _PEAX:
                        nt = max(nt - 1, 0)
                if (accum + nt) * NT <= idx:
                    accum += nt
                    batch += Int32(1)
                else:
                    keep = cutlass.Boolean(False)
            elif cutlass.const_expr(_PEA > 0) and phase == 0:
                phase = Int32(2)
                base = base + accum * NT
                idx = self._lin - base
                batch = Int32(0)
                accum = Int32(0)
            else:
                if phase == (2 if cutlass.const_expr(_PEA > 0) else 0) and cutlass.const_expr(_PSTOP == 0):
                    phase = Int32(1)
                    base = base + accum * NT
                    idx = self._lin - base
                    if cutlass.const_expr(_PSPL > 0):
                        idx, hf = self._pmap(base)
                    batch = Int32(0)
                    accum = Int32(0)
                else:
                    keep = cutlass.Boolean(False)
        self._batch = batch
        self._accum = accum
        self._phase = phase
        self._base = base
        is_valid = batch < num_batches
        moff = Int32(0)
        if cutlass.const_expr(_PEA > 0):
            if phase == 2 and batch < _PEAX:
                moff = Int32(1)
        cm = idx % NT
        if cutlass.const_expr(_PCL > 1):
            cm = cm * _PCL + cute.arch.block_idx()[0] % _PCL
        coord = (cm, idx // NT - accum + moff, batch, (phase & 1) + hf * 4)
        return WorkTileInfo(coord, is_valid)

    @cute.jit
    def _get_tab(self):
        # round 7 bound (P_ORD=2): tile order from a host-built table stored at masked_m + 64 (packed m | n<<8 | l<<16 |
        # phase<<24, -1 = end); one load per tile, so the order is tested without scheduler cost.
        num_batches = self.params.masked_m.shape[0]
        tb = cute.make_tensor(self.params.masked_m.iterator + 64, cute.make_layout(32768))
        v = tb[self._lin]
        batch = Int32(num_batches)
        if v >= 0:
            batch = (v >> 16) & 255
        self._batch = batch
        cm = v & 255
        if cutlass.const_expr(_PCL > 1):
            cm = cm * _PCL + cute.arch.block_idx()[0] % _PCL
        coord = (cm, (v >> 8) & 255, batch, (v >> 24) & 1)
        return WorkTileInfo(coord, batch < num_batches)

    _get = _get_tab if _PORD == 2 else (_get_ord if _PORD else _get_rr)

    @cute.jit
    def _pmap(self, n1):
        # Lane P round 3: balanced static FC2 assignment. CTA c took ceil((N1-c)/G) FC1 tiles; CTAs c >= N1 % G took one
        # fewer and get the first X FC2 units (X units = one FC1 tile of k-blocks). Monotone per CTA, covers all units.
        G = self.num_persistent_clusters
        lin = self._lin
        X = 2 if cutlass.const_expr(_PSPL == 1) else 4
        c = lin % G
        r = n1 % G
        g7 = (G - r) % G
        j = lin // G - (n1 - c + G - 1) // G
        u = X * g7 + c + G * j
        if r != 0:
            if c >= r:
                if j < X:
                    u = (c - r) + g7 * j
                else:
                    u = X * g7 + c + G * (j - X)
        t = u
        h = Int32(0)
        if cutlass.const_expr(_PSPL == 2):
            t = u >> 1
            h = u & 1
        return t, h

    @dsl_user_op
    def get_current_work(self, dsm_pending_packed=None, dsm_counter=None, num_c_stage=None, *, loc=None, ip=None):
        return self._get(), dsm_pending_packed

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self._get()

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        self._lin += Int32(advance_count) * Int32(self.num_persistent_clusters)
        self._num_tiles_executed += Int32(1)

    @property
    def num_tiles_executed(self):
        return self._num_tiles_executed


class MaskedSchedulerParams:
    def __init__(
        self,
        masked_m: cute.Tensor,
        dst_signals: Optional[cute.Pointer],
        c: cute.Tensor,
        c_tiler: Tuple[int, int],
        cluster_shape_mnk: cute.Shape,
        is_swap_ab: bool = False,
        *,
        loc=None,
        ip=None,
    ):
        if cluster_shape_mnk[2] != 1:
            raise ValueError(f"unsupported cluster_shape_k {cluster_shape_mnk[2]}")

        gc = cute.zipped_divide(c, tiler=c_tiler)
        problem_shape_ntile_mnl = gc[(0, (None, None, None))].shape
        self.masked_m = masked_m
        self.dst_signals = dst_signals
        self.c = c
        self.c_tiler = c_tiler
        self.problem_shape_ntile_mnl = problem_shape_ntile_mnl
        # cluster_shape_mnk is kept for reconstruction
        self._cluster_shape_mnk = cluster_shape_mnk
        self.cluster_shape_mn = cluster_shape_mnk[:2]
        self.is_swap_ab = is_swap_ab
        self._loc = loc

        self.problem_layout_ncluster_mnl = cute.make_layout(
            cute.ceil_div(
                self.problem_shape_ntile_mnl, cluster_shape_mnk[:2], loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.masked_m,
            self.dst_signals,
            self.c,
            self.c_tiler,
            self._cluster_shape_mnk,
            self.is_swap_ab,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.masked_m,
                self.dst_signals,
                self.c,
                self.c_tiler,
                self._cluster_shape_mnk,
                self.is_swap_ab,
            ],
            self._values_pos,
            strict=True,
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return MaskedSchedulerParams(*(tuple(obj_list)), loc=self._loc)

    @dsl_user_op
    def get_grid_shape(
        self, max_active_clusters: Int32, *, loc=None, ip=None
    ) -> Tuple[Integer, Integer, Integer]:
        total_tile_clusters = cute.size(
            self.problem_layout_ncluster_mnl, loc=loc, ip=ip
        )
        num_persistent_clusters = min(max_active_clusters, total_tile_clusters)

        return (*self.cluster_shape_mn, num_persistent_clusters)


class MaskedScheduler:
    def __init__(
        self,
        params: MaskedSchedulerParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        current_batch_idx: Int32,
        accum_tile_m: Int32,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
    ):
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self._current_batch_idx = current_batch_idx
        self._accum_tile_m = accum_tile_m
        self.cta_id_in_cluster = cta_id_in_cluster
        self._num_tiles_executed = num_tiles_executed

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.num_persistent_clusters)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self._current_batch_idx))
        values.extend(extract_mlir_values(self._accum_tile_m))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "MaskedScheduler":
        assert len(values) == 8
        new_num_persistent_clusters = new_from_mlir_values(
            self.num_persistent_clusters, [values[0]]
        )
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[1]]
        )
        new_current_batch_idx = new_from_mlir_values(
            self._current_batch_idx, [values[2]]
        )
        new_accum_tile_m = new_from_mlir_values(self._accum_tile_m, [values[3]])
        new_cta_id_in_cluster = new_from_mlir_values(
            self.cta_id_in_cluster, values[4:7]
        )
        new_num_tiles_executed = new_from_mlir_values(
            self._num_tiles_executed, [values[7]]
        )
        return MaskedScheduler(
            self.params,
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_current_batch_idx,
            new_accum_tile_m,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
        )

    # called by host
    @dsl_user_op
    @staticmethod
    def create(
        params: MaskedSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        *,
        loc=None,
        ip=None,
    ):
        params = params

        # Calculate the number of persistent clusters by dividing the total grid size
        # by the number of CTAs per cluster
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(
            params.cluster_shape_mn, loc=loc, ip=ip
        )

        bidx, bidy, bidz = block_idx

        # Initialize workload index equals to the cluster index in the grid
        current_work_linear_idx = Int32(bidz)
        current_batch_idx = Int32(0)
        accum_tile_m = Int32(0)

        # CTA id in the cluster
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        # Initialize number of tiles executed to zero
        num_tiles_executed = Int32(0)
        return MaskedScheduler(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            current_batch_idx,
            accum_tile_m,
            cta_id_in_cluster,
            num_tiles_executed,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: MaskedSchedulerParams,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Integer, Integer, Integer]:
        return params.get_grid_shape(max_active_clusters, loc=loc, ip=ip)

    # private method
    @cute.jit
    def _get_current_work_for_linear_idx(
        self,
        current_work_linear_idx: Int32,
        dsm_pending_packed: Optional[Uint64],
        dsm_counter: Optional[Uint8],
        num_c_stage: Optional[int] = None,
    ) -> Tuple[WorkTileInfo, Optional[Uint64]]:
        # is_valid = current_work_linear_idx < cute.size(
        #     self.params.problem_layout_ncluster_mnl, loc=loc, ip=ip
        # )
        is_swap_ab = self.params.is_swap_ab
        num_tiles_n = self.params.problem_shape_ntile_mnl[
            0 if cutlass.const_expr(is_swap_ab) else 1
        ]
        accum_tile_m = self._accum_tile_m
        batch_idx = self._current_batch_idx
        num_batches = self.params.masked_m.shape[0]

        # NOTE: `masked_m[batch_idx]` must only be read under a control-flow
        # guard that `batch_idx` is in bounds. A combined condition like
        # `masked_m[batch_idx] ... and batch_idx < num_batches` is NOT safe:
        # dynamic `and` in the DSL may evaluate both operands (no Python
        # short-circuit), so `masked_m[num_batches]` could be read
        # out-of-bounds once all batches are consumed, causing illegal
        # memory access (CUDA error 700).
        keep_running = batch_idx < num_batches
        while keep_running:
            num_tiles_m_cur = cute.ceil_div(
                self.params.masked_m[batch_idx],
                self.params.c_tiler[1 if cutlass.const_expr(is_swap_ab) else 0],
            )
            if (accum_tile_m + num_tiles_m_cur) * num_tiles_n <= (
                current_work_linear_idx
            ):
                if cutlass.const_expr(
                    (dsm_pending_packed is not None)
                    and (self.params.dst_signals is not None)
                ):
                    dsm_pending_packed = with_byte(
                        dsm_pending_packed,
                        index=batch_idx,
                        value=dsm_counter + (num_c_stage - 1),
                    )

                accum_tile_m += num_tiles_m_cur
                batch_idx += Int32(1)
                keep_running = batch_idx < num_batches
            else:
                keep_running = cutlass.Boolean(False)

        self._accum_tile_m = accum_tile_m
        self._current_batch_idx = batch_idx

        is_valid = batch_idx < num_batches
        if is_valid:
            is_valid = (
                accum_tile_m
                + cute.ceil_div(
                    self.params.masked_m[batch_idx],
                    self.params.c_tiler[1 if cutlass.const_expr(is_swap_ab) else 0],
                )
            ) * num_tiles_n > current_work_linear_idx

        # cur_cluster_coord = self.params.problem_layout_ncluster_mnl.get_hier_coord(
        #     current_work_linear_idx, loc=loc, ip=ip
        # )
        if cutlass.const_expr(is_swap_ab):
            cur_cluster_coord = (
                current_work_linear_idx % num_tiles_n,
                current_work_linear_idx // num_tiles_n - accum_tile_m,
                batch_idx,
            )
        else:
            cur_cluster_coord = (
                current_work_linear_idx // num_tiles_n - accum_tile_m,
                current_work_linear_idx % num_tiles_n,
                batch_idx,
            )

        # cur_tile_coord is a tuple of i32 values
        cur_tile_coord = tuple(
            Int32(x) * Int32(z) + Int32(y)
            for x, y, z in zip(
                cur_cluster_coord,
                self.cta_id_in_cluster,
                (*self.params.cluster_shape_mn, Int32(1)),
                strict=True,
            )
        )

        return WorkTileInfo(cur_tile_coord, is_valid), dsm_pending_packed

    @dsl_user_op
    def get_current_work(
        self,
        dsm_pending_packed: Optional[Uint64] = None,
        dsm_counter: Optional[Uint8] = None,
        num_c_stage: Optional[int] = None,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[WorkTileInfo, Optional[Uint64]]:
        return self._get_current_work_for_linear_idx(
            self._current_work_linear_idx,
            dsm_pending_packed=dsm_pending_packed,
            dsm_counter=dsm_counter,
            num_c_stage=num_c_stage,
        )

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        tile_info, _ = self.get_current_work(loc=loc, ip=ip)
        return tile_info

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        self._current_work_linear_idx += Int32(advance_count) * Int32(
            self.num_persistent_clusters
        )
        self._num_tiles_executed += Int32(1)

    @property
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed


@dsl_user_op
def make_fused_smem_layout_epi(
    epi_dtype: Type[cutlass.Numeric],
    epi_layout: utils.LayoutEnum,
    epi_tile: cute.Tile,
    epi_stage: int,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    epilog_shape = cute.product_each(
        cute.shape(epi_tile, loc=loc, ip=ip), loc=loc, ip=ip
    )
    pad = 8  # shift each row to a different smem bank
    c_smem_layout_atom = cute.make_composed_layout(
        cute.make_swizzle(0, 4, 3),
        0,
        cute.make_layout((epi_tile[0].shape, 8), stride=(1, epi_tile[0].shape + pad)),
        loc=loc,
        ip=ip,
    )

    epilog_shape = cute.append(epilog_shape, epi_stage, loc=loc, ip=ip)
    epi_smem_layout_staged = cute.tile_to_shape(
        c_smem_layout_atom,
        epilog_shape,
        order=((1, 0, 2) if not epi_layout.is_n_major_c() else (0, 1, 2)),
        loc=loc,
        ip=ip,
    )

    return epi_smem_layout_staged


def l_prefetch_l2(addr, size, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(size).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def p_fence_proxy_async_global(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "fence.proxy.async.global;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def p_ld_vol_shared(addr, *, loc=None, ip=None):
    return Int32(llvm.inline_asm(T.i32(), [Int32(addr).ir_value(loc=loc, ip=ip)], "ld.volatile.shared::cta.b32 $0, [$1];", "=r,r",
                                 has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def p_ld_relaxed(addr, *, loc=None, ip=None):
    return Int32(llvm.inline_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip)], "ld.relaxed.gpu.global.b32 $0, [$1];", "=r,l",
                                 has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def p_mkpol(kind, *, loc=None, ip=None):
    """L2 cache policy for TMA (kind: evict_first / evict_last / evict_normal)."""
    return Int64(llvm.inline_asm(T.i64(), [], "createpolicy.fractional.L2::" + kind + ".b64 $0, 1.0;", "=l",
                                 has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


def p_unig(v):
    """gate-warp poll: uniform only when the whole gate warp runs the walk (P_GL=32)."""
    return cute.arch.make_warp_uniform(v) if (_PUNI and _PGL == 32) else v


def p_uni(v):
    """lane P: mark a polled value warp-uniform so ptxas keeps the role's loop on the uniform datapath."""
    return cute.arch.make_warp_uniform(v) if _PUNI else v


@dsl_user_op
def p_st_vol_global(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
                    "st.volatile.global.b32 [$0], $1;", "l,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def p_st_vol_shared(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
                    "st.volatile.shared::cta.b32 [$0], $1;", "r,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def p_nanosleep(ns, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(ns).ir_value(loc=loc, ip=ip)], "nanosleep.u32 $0;", "r", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def p_ld_acq_shared(addr, *, loc=None, ip=None):
    return Int32(llvm.inline_asm(T.i32(), [Int32(addr).ir_value(loc=loc, ip=ip)], "ld.acquire.cta.shared::cta.b32 $0, [$1];", "=r,r",
                                 has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def p_st_rel_shared(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
                    "st.release.cta.shared::cta.b32 [$0], $1;", "r,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def p_red_release(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
                    "red.release.gpu.global.add.s32 [$0], $1;", "l,r", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def p_atom_add(addr, v, *, loc=None, ip=None):
    return Int32(llvm.inline_asm(T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
                                 "atom.add.relaxed.gpu.global.s32 $0, [$1], $2;", "=r,l,r",
                                 has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def p_red_bf16x8(dst_addr, r0, r1, r2, r3, *, loc=None, ip=None):
    """16 B register -> gmem vector bf16x2 reduction (lane I b6vp finalize)."""
    llvm.inline_asm(
        None,
        [Int64(dst_addr).ir_value(loc=loc, ip=ip)] + [cutlass.Uint32(r).ir_value(loc=loc, ip=ip) for r in (r0, r1, r2, r3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};" if _PNORED else "red.global.add.noftz.v4.bf16x2 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cp_reduce_bf16_add(dst_addr, src_addr, size, *, loc=None, ip=None):
    """Async bulk copy-reduce (add) from shared memory to global memory.

    PTX instruction: cp.reduce.async.bulk.global.shared::cta.bulk_group.add.bf16.noftz [dstMem], [srcMem], size;

    Performs an asynchronous bulk reduction operation that atomically adds bf16 values
    from shared memory (CTA scope) to global memory. The operation is part of a bulk group.

    Args:
        dst_addr: Destination global memory address (pointer)
        src_addr: Source shared memory address (pointer)
        size: Size in bytes of the data to reduce (must be a multiple of 4)
    """
    llvm.inline_asm(
        None,
        [
            dst_addr.toint(loc=loc, ip=ip).ir_value(),
            src_addr.toint(loc=loc, ip=ip).ir_value(),
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.bf16.noftz [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def cp_reduce_f32_add(dst_addr, src_addr, size, *, loc=None, ip=None):
    """Async bulk copy-reduce (add) from shared memory to global memory.

    PTX instruction: cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [dstMem], [srcMem], size;

    Performs an asynchronous bulk reduction operation that atomically adds f32 values
    from shared memory (CTA scope) to global memory. The operation is part of a bulk group.

    Args:
        dst_addr: Destination global memory address (pointer)
        src_addr: Source shared memory address (pointer)
        size: Size in bytes of the data to reduce (must be a multiple of 4)
    """
    llvm.inline_asm(
        None,
        [
            dst_addr.toint(loc=loc, ip=ip).ir_value(),
            src_addr.toint(loc=loc, ip=ip).ir_value(),
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        """
        cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;
        """,
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )



@dsl_user_op
def h_st_zero16(addr, *, loc=None, ip=None):
    """st.global.v4.b32 of zeros (lane I: FC2 output zeroing folded into the FC1 epilogue warps)."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 z; mov.b32 z, 0; st.global.v4.b32 [$0], {z, z, z, z}; }",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

"""
This example provides an experimental implementation of the SM100 batched dense blockscaled GEMM kernel, please note that the APIs and implementation details related to this kernel may change in future releases.

A high-performance persistent batched dense blockscaled GEMM example for the NVIDIA Blackwell SM100 architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M") for MXF8 input type and can only be row-major("K") for MXF4/NVF4 input type
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K") for MXF8 input type and can only be row-major("K") for MXF4/NVF4 input type
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk, which has M×ceil_div(K, sf_vec_size)×L elements respectively
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk, which has N×ceil_div(K, sf_vec_size)×L elements respectively

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp:
    - Load scale factor A/B from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp instruction.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp:
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Type convert C matrix to output type.
    - Optionally store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations,
      or directly store C matrix from registers (RMEM) to global memory (GMEM) without TMA operations.

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Input arguments to this example is shown below:

.. code-block:: bash

    python examples/blackwell/dense_blockscaled_gemm_persistent.py            \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/dense_blockscaled_gemm_persistent.py        \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16        \
      --c_dtype Float16                                                        \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1                            \
      --mnkl 8192,8192,1024,1                                                  \
      --warmup_iterations 1 --iterations 10 --skip_ref_check


Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 128 or 256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.
"""


class Sm100BlockScaledPersistentDenseGemmKernel:
    """This class implements batched matrix multiplication (C = A x SFA x B x SFB) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 128/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

    Example:
        >>> gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        ...     sf_vec_size=16,
        ...     mma_tiler_mn=(256, 128),
        ...     cluster_shape_mn=(2, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor, max_active_clusters, stream)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        sm_version: str,
        num_ranks: int = 0,
        is_combine_fusion: bool = False,
        is_swap_ab: bool = False,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator, always set to Float32
            - sf_vec_size: Scalefactor A/B vector size.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        """
        supported_sm_versions = ["sm_100", "sm_103", "sm_107"]
        assert sm_version in supported_sm_versions, (
            f"{supported_sm_versions} are the only supported SM versions for cute-dsl backend, but encountered {sm_version}"
        )

        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.num_ranks = num_ranks
        self.is_combine_fusion = is_combine_fusion
        self.is_swap_ab = is_swap_ab

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.tmab_warp_id = 6  # lane H split rings (H_SCHED=3): dedicated token (B) producer warp
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        ) + (32 if (_HS == 3 or _PG) else 0)
        self.p_gate_warp_id = 6  # lane P gate warp (acquires FC2 gates ahead of the TMA producer)
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.smem_capacity = utils.get_smem_capacity_in_bytes(sm_version)
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    def j_mk_sched(self, producer, epi, tile_sched_params, j_ctx):
        if cutlass.const_expr(_JD):
            ex = (tile_sched_params.masked_m, self.cta_tile_shape_mnk[1]) if (epi and _HS == 2) else None
            return DynScheduler(j_ctx, producer, Int32(0), Int32(0), ex)
        return (FusedScheduler if _PF else (UnitScheduler if _HS else MaskedScheduler)).create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )

    @cute.jit
    def p_zero_share(self, p_out, p_gates, tidx):
        pz_base = Int64(p_out.iterator.toint())
        pz_n = p_gates[2 * _PNG + 1]
        pgx, pgy, pgz = cute.arch.grid_dim()
        pbx, pby, pbz = cute.arch.block_idx()
        pz_thr = pgx * pgy * pgz * 128
        pzi = ((pbz * pgy + pby) * pgx + pbx) * 128 + tidx
        while pzi < pz_n:
            h_st_zero16(pz_base + Int64(pzi) * Int64(16))
            pzi += pz_thr
        cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=128)
        if tidx == 0:
            j_fence_gpu()
            p_atom_add(Int64(p_gates.iterator.toint()) + Int64(4 * (2 * _PNG + 7)), Int32(1))

    @cute.jit
    def j_trace(self, j_order, bidz, slot):
        if cutlass.const_expr(_JT):
            if cute.arch.lane_idx() == 0:
                tt = j_gtimer()
                j_order[8192 + bidz * 64 + slot] = tt

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Compute mma instruction shapes
        mma_inst_bits_k = 256
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_bits_k // self.a_dtype.width,
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk_sfb = (
            self.mma_inst_shape_mnk[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mnk[1], 128),
            self.mma_inst_shape_mnk[2],
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_tile_k = _GK
        self.mma_tiler = (
            self.mma_inst_shape_mnk[0],
            self.mma_inst_shape_mnk[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mnk_sfb[0],
            self.mma_inst_shape_mnk_sfb[1],
            self.mma_inst_shape_mnk_sfb[2] * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.a_major_mode,
            self.b_dtype,
            self.b_major_mode,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            self.is_combine_fusion,
        )

        self.num_b_stage = self.num_ab_stage
        if _HS == 3:
            a1 = cute.size_in_bytes(self.a_dtype, sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, 1)) + cute.size_in_bytes(
                self.sf_dtype, blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, 1))
            b1 = cute.size_in_bytes(self.b_dtype, sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, 1)) + cute.size_in_bytes(
                self.sf_dtype, blockscaled_utils.make_smem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, 1))
            self.num_b_stage = _HBST
            self.num_ab_stage = (self.smem_capacity - 2048 - 64 * (_HEPI_COLS + 1) * 4 - _HBST * b1) // a1
            if _GAB:
                self.num_ab_stage = _GAB
            print('[H] split rings', dict(a_stage_bytes=a1, num_a_stage=self.num_ab_stage, b_stage_bytes=b1, num_b_stage=_HBST,
                  a_in_flight=a1 * self.num_ab_stage, smem=a1 * self.num_ab_stage + b1 * _HBST + 2048 + 64 * (_HEPI_COLS + 1) * 4), flush=True)

        print('[STG] ab', self.num_ab_stage, 'acc', self.num_acc_stage, 'c', self.num_c_stage, 'mod', __name__, flush=True)
        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_b_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_b_stage,
        )
        if self.is_combine_fusion:
            self.c_smem_layout_staged = make_fused_smem_layout_epi(
                self.c_dtype,
                self.c_layout,
                self.epi_tile,
                self.num_c_stage,
            )
        else:
            self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.c_dtype,
                self.c_layout,
                self.epi_tile,
                self.num_c_stage,
            )

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        masked_m_tensor: cute.Tensor,
        dst_signals: Optional[cute.Pointer],
        alpha_tensor: Optional[cute.Tensor],
        topk_weights: Optional[cute.Tensor],
        idx_src_info: Optional[cute.Tensor],
        rank_src_info: Optional[cute.Tensor],
        out_ptrs: Optional[cute.Tensor],
        barrier_flag_local: Optional[cute.Pointer],
        barrier_flag_multicast: Optional[cute.Pointer],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        h_out: Optional[cute.Tensor] = None,
        h_sf: Optional[cute.Tensor] = None,
        h_gs: Optional[cute.Tensor] = None,
        j_order: Optional[cute.Tensor] = None,
        j_ctr: Optional[cute.Tensor] = None,
        j_ws: Optional[cute.Tensor] = None,
        p_args=None,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a_tensor: Input tensor A
        :type a_tensor: cute.Tensor
        :param b_tensor: Input tensor B
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor tensor A
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor tensor B
        :type sfb_tensor: cute.Tensor
        :param c_tensor: Output tensor C
        :type c_tensor: cute.Tensor
        :param masked_m_tensor: Masked layout tensor M
        :type masked_m_tensor: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param alpha_tensor: Optional 1D tensor of shape (l,) containing per-batch scaling factors.
        :type alpha_tensor: cute.Tensor
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.h_b_tx = (b_copy_size + sfb_copy_size) * atom_thr_size
        self.num_tma_load_bytes = (
            a_copy_size + sfa_copy_size + (0 if (_GNOB or _HS == 3) else b_copy_size + sfb_copy_size)
        ) * atom_thr_size

        # Lane J variable-N: 32-row B boxes + N=32/64 MMA variants on prefixes of the 128-row B buffer
        j_tm32 = j_tm64 = j_tma_atom_b32 = j_tma_tensor_b32 = j_b32_staged = j_b64_staged = None
        if cutlass.const_expr(_JV):
            j_tm32 = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, self.cta_group, (self.mma_tiler[0], 32))
            j_tm64 = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, self.cta_group, (self.mma_tiler[0], 64))
            mt32 = (self.mma_tiler[0], 32, self.mma_tiler[2])
            mt64 = (self.mma_tiler[0], 64, self.mma_tiler[2])
            nsub = self.mma_tiler[1] // 32
            j_b32_staged = sm100_utils.make_smem_layout_b(j_tm32, mt32, self.b_dtype, nsub * self.num_ab_stage)
            j_b64_staged = sm100_utils.make_smem_layout_b(j_tm64, mt64, self.b_dtype, (self.mma_tiler[1] // 64) * self.num_ab_stage)
            b32_one = cute.slice_(j_b32_staged, (None, None, None, 0))
            j_tma_atom_b32, j_tma_tensor_b32 = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, b_tensor, b32_one, mt32, j_tm32, self.cluster_layout_vmnk.shape)
            self.j_b32_bytes = cute.size_in_bytes(self.b_dtype, b32_one) * atom_thr_size
            self.j_base_tx = (a_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size
            self.p_aw_tx = (a_copy_size + sfa_copy_size) * atom_thr_size
            self.j_sfb_tx = sfb_copy_size * atom_thr_size
            print('[J] varN', dict(b32_bytes=self.j_b32_bytes, base_tx=self.j_base_tx, full_tx=self.num_tma_load_bytes,
                  b_cosize=cute.cosize(self.b_smem_layout_staged.outer), b32_cosize=cute.cosize(j_b32_staged.outer)), flush=True)

        p_kargs = [None] * 13
        if cutlass.const_expr(_PF):
            a2_tensor, b2_tensor, sfa2_raw, sfb2_raw, alpha2_t, p_out_t, p_perm_t, p_rw_t, p_gates_t = p_args
            sfa2_tensor = cute.make_tensor(sfa2_raw.iterator, blockscaled_utils.tile_atom_to_shape_SF(a2_tensor.shape, self.sf_vec_size))
            sfb2_tensor = cute.make_tensor(sfb2_raw.iterator, blockscaled_utils.tile_atom_to_shape_SF(b2_tensor.shape, self.sf_vec_size))
            tma_atom_a2, tma_tensor_a2 = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, a2_tensor, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
            tma_atom_sfa2, tma_tensor_sfa2 = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op, sfa2_tensor, sfa_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16)
            tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op, sfb2_tensor, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb, self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16)
            tma_atom_b32_2, tma_tensor_b32_2 = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, b2_tensor, b32_one, mt32, j_tm32, self.cluster_layout_vmnk.shape)
            p_kargs = [tma_atom_a2, tma_tensor_a2, tma_atom_sfa2, tma_tensor_sfa2, tma_atom_b32_2, tma_tensor_b32_2,
                       tma_atom_sfb2, tma_tensor_sfb2, alpha2_t, p_out_t, p_perm_t, p_rw_t, p_gates_t]

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = c_tensor
        if cutlass.const_expr(not self.is_combine_fusion and not _HF):
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c_tensor,
                epi_smem_layout,
                self.epi_tile,
            )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            masked_m_tensor,  # add masked layout
            dst_signals,
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.is_swap_ab,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            b_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_b_stage]
            b_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_b_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            j_full: cute.struct.MemRange[cutlass.Int64, _JQ]
            j_empty: cute.struct.MemRange[cutlass.Int64, _JQ]
            j_buf: cute.struct.MemRange[cutlass.Int32, _JQ]
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            if cutlass.const_expr(not _HF):
                sC: cute.struct.Align[
                    cute.struct.MemRange[
                        self.c_dtype,
                        cute.cosize(self.c_smem_layout_staged.outer),
                    ],
                    self.buffer_align_bytes,
                ]
            else:
                sEpi: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, max(64 * (_HEPI_COLS + 1) * 4, _HEPI_COLS * 272)], 128]  # FC1 f32 64x33 | FC2 bf16 32x136
                sW: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 2 * _HEPI_COLS], 16]
                sR: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 2 * _HEPI_COLS], 16]
                pFlags: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            if cutlass.const_expr(self.is_combine_fusion):
                # Prefetch buffer for out_ptrs (one Int64 per M-row in the MMA tile)
                combine_out_ptrs: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Int64,
                        self.mma_inst_shape_mnk[1],
                    ],
                    16,
                ]
                # Prefetch buffer for topk_weights (one Float32 per M-row in the MMA tile)
                combine_topk: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Float32,
                        self.mma_inst_shape_mnk[1],
                    ],
                    16,
                ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            c_tensor if self.is_combine_fusion else tma_tensor_c,
            alpha_tensor,
            topk_weights,
            idx_src_info,
            rank_src_info,
            out_ptrs,
            barrier_flag_local,
            barrier_flag_multicast,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            h_out,
            h_sf,
            h_gs,
            j_order,
            j_ctr,
            j_ws,
            j_tm32,
            j_tm64,
            j_tma_atom_b32,
            j_tma_tensor_b32,
            j_b32_staged,
            j_b64_staged,
            *p_kargs,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=bool(_HPDLW),
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        alpha: Optional[cute.Tensor],
        topk_weights: Optional[cute.Tensor],
        idx_src_info: Optional[cute.Tensor],
        rank_src_info: Optional[cute.Tensor],
        out_ptrs: Optional[cute.Tensor],
        barrier_flag_local: Optional[cute.Tensor],
        barrier_flag_multicast: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: MaskedSchedulerParams,
        h_out: Optional[cute.Tensor],
        h_sf: Optional[cute.Tensor],
        h_gs: Optional[cute.Tensor],
        j_order: Optional[cute.Tensor],
        j_ctr: Optional[cute.Tensor],
        j_ws: Optional[cute.Tensor],
        j_tm32: Optional[cute.TiledMma],
        j_tm64: Optional[cute.TiledMma],
        j_tma_atom_b32: Optional[cute.CopyAtom],
        j_tma_tensor_b32: Optional[cute.Tensor],
        j_b32_staged: Optional[cute.ComposedLayout],
        j_b64_staged: Optional[cute.ComposedLayout],
        tma_atom_a2: Optional[cute.CopyAtom],
        mA2: Optional[cute.Tensor],
        tma_atom_sfa2: Optional[cute.CopyAtom],
        mSFA2: Optional[cute.Tensor],
        tma_atom_b32_2: Optional[cute.CopyAtom],
        mB32_2: Optional[cute.Tensor],
        tma_atom_sfb2: Optional[cute.CopyAtom],
        mSFB2: Optional[cute.Tensor],
        alpha2: Optional[cute.Tensor],
        p_out: Optional[cute.Tensor],
        p_perm: Optional[cute.Tensor],
        p_rw: Optional[cute.Tensor],
        p_gates: Optional[cute.Tensor],
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if cutlass.const_expr(_HPDLT):
            cute.arch.griddepcontrol_launch_dependents()  # lane I: let FC2 (PDL) launch onto SMs as FC1 CTAs retire

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            if cutlass.const_expr(_JV):
                cpasync.prefetch_descriptor(j_tma_atom_b32)
            if cutlass.const_expr(_PF):
                cpasync.prefetch_descriptor(tma_atom_a2)
                cpasync.prefetch_descriptor(tma_atom_sfa2)
                cpasync.prefetch_descriptor(tma_atom_b32_2)
                cpasync.prefetch_descriptor(tma_atom_sfb2)
            if cutlass.const_expr(tma_atom_c is not None):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.ptr
        tmem_holding_buf = storage.tmem_holding_buf.ptr

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage // _PK,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes * _PK,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        b_pipeline = None
        if cutlass.const_expr(_HS == 3):
            b_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.b_full_mbar_ptr.data_ptr(),
                num_stages=self.num_b_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer),
                tx_count=self.h_b_tx,
                cta_layout_vmnk=cluster_layout_vmnk,
            )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = (
            self.threads_per_warp
            * len(self.epilog_warp_id)
            * (2 if use_2cta_instrs else 1)
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Lane J dynamic scheduler ring
        j_full = storage.j_full.data_ptr()
        j_empty = storage.j_empty.data_ptr()
        j_buf = storage.j_buf.get_tensor(cute.make_layout(_JQ))
        if cutlass.const_expr(_JD == 1):
            if warp_idx == self.tma_warp_id:
                with cute.arch.elect_one():
                    for ji in cutlass.range_constexpr(_JQ):
                        cute.arch.mbarrier_init(j_full + ji, 1)
                        cute.arch.mbarrier_init(j_empty + ji, self.threads_per_warp * (1 + len(self.epilog_warp_id)))

        if cutlass.const_expr(_PEA > 0 and _JD == 0):
            if warp_idx == self.tma_warp_id:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(j_full, 1)  # lane P: early-weight barrier (j_full[0] unused without J_DYN)
        j_ctx = (j_order, j_ctr, j_buf, j_full, j_empty)

        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )
        cute.arch.mbarrier_init_fence()

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        if cutlass.const_expr(not _HF):
            sC = storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
        else:
            sEpi = cute.make_tensor(cute.recast_ptr(storage.sEpi.data_ptr(), dtype=cutlass.Float32),
                                    cute.make_layout((64, _HEPI_COLS), stride=(_HEPI_COLS + 1, 1)))
            sEpi2 = cute.make_tensor(cute.recast_ptr(storage.sEpi.data_ptr(), dtype=cutlass.BFloat16),
                                     cute.make_layout((_HEPI_COLS, 128), stride=(136, 1)))
            sW = storage.sW.get_tensor(cute.make_layout((_HEPI_COLS, 2)))
            sR = storage.sR.get_tensor(cute.make_layout((_HEPI_COLS, 2)))
            p_flags = storage.pFlags.get_tensor(cute.make_layout(4))
            p_flag_addr = Int32(p_flags.iterator.toint())
            if tidx == 0:
                p_flags[0] = Int32(0)
        if cutlass.const_expr(self.is_combine_fusion):
            smem_out_ptrs = storage.combine_out_ptrs.get_tensor(
                cute.make_layout(self.mma_inst_shape_mnk[1])
            )
            smem_topk = storage.combine_topk.get_tensor(
                cute.make_layout(self.mma_inst_shape_mnk[1])
            )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_block_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        if cutlass.const_expr(_JV):
            sB32 = storage.sB.get_tensor(j_b32_staged.outer, swizzle=j_b32_staged.inner)
            sB64 = storage.sB.get_tensor(j_b64_staged.outer, swizzle=j_b64_staged.inner)
            gB32 = cute.local_tile(j_tma_tensor_b32, (32, self.mma_tiler[2]), (None, None, None))
            tCgB32 = j_tm32.get_slice(mma_tile_coord_v).partition_B(gB32)
            tBsB32, tBgB32 = cpasync.tma_partition(
                j_tma_atom_b32,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sB32, 0, 3),
                cute.group_modes(tCgB32, 0, 3),
            )
        if cutlass.const_expr(_PF):
            gA2_mkl = cute.local_tile(mA2, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
            gSFA2_mkl = cute.local_tile(mSFA2, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
            gSFB2_nkl = cute.local_tile(mSFB2, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
            k_block_cnt2 = cute.size(gA2_mkl, mode=[3])
            tAsA2, tAgA2 = cpasync.tma_partition(tma_atom_a2, block_in_cluster_coord_vmnk[2], a_cta_layout,
                                                 cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA2_mkl), 0, 3))
            tAsSFA2, tAgSFA2 = cpasync.tma_partition(tma_atom_sfa2, block_in_cluster_coord_vmnk[2], sfa_cta_layout,
                                                     cute.group_modes(sSFA, 0, 3), cute.group_modes(thr_mma.partition_A(gSFA2_mkl), 0, 3))
            tAsSFA2 = cute.filter_zeros(tAsSFA2)
            tAgSFA2 = cute.filter_zeros(tAgSFA2)
            tBsSFB2, tBgSFB2 = cpasync.tma_partition(tma_atom_sfb2, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout,
                                                     cute.group_modes(sSFB, 0, 3), cute.group_modes(thr_mma_sfb.partition_B(gSFB2_nkl), 0, 3))
            tBsSFB2 = cute.filter_zeros(tBsSFB2)
            tBgSFB2 = cute.filter_zeros(tBgSFB2)
            gB32_2 = cute.local_tile(mB32_2, (32, self.mma_tiler[2]), (None, None, None))
            tBsB32_2, tBgB32_2 = cpasync.tma_partition(tma_atom_b32_2, block_in_cluster_coord_vmnk[1], b_cta_layout,
                                                       cute.group_modes(sB32, 0, 3),
                                                       cute.group_modes(j_tm32.get_slice(mma_tile_coord_v).partition_B(gB32_2), 0, 3))

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        if cutlass.const_expr(_JV):
            tCrB32 = j_tm32.make_fragment_B(sB32)
            tCrB64 = j_tm64.make_fragment_B(sB64)
            j_acc32_layout = j_tm32.make_fragment_C(j_tm32.partition_shape_C((self.mma_tiler[0], 32))).layout
            j_acc64_layout = j_tm64.make_fragment_C(j_tm64.partition_shape_C((self.mma_tiler[0], 64))).layout

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
            )
        if cutlass.const_expr(_PEA > 0):
            if warp_idx == self.tma_warp_id:
                # lane P: weights do not depend on routing. FC1 part A puts tile (expert c // NT, weight tile c % NT, m=0)
                # first on CTA c, so its first _PEA A+SFA k-blocks go into ring stages 0.._PEA-1 before the PDL wait.
                e_c = cute.arch.block_idx()[2]
                e_l = e_c // Int32(32)
                e_m = e_c % Int32(32)
                if e_l < Int32(_PEAX):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(j_full, _PEA * self.p_aw_tx)
                    e_cp = {}
                    if cutlass.const_expr(_PEVF):
                        e_cp = {"cache_policy": p_mkpol("evict_first")}
                    e_ga = tAgA[(None, e_m, None, e_l)]
                    e_gs = tAgSFA[(None, e_m, None, e_l)]
                    for e_s in cutlass.range_constexpr(_PEA):
                        cute.copy(tma_atom_a, e_ga[(None, e_s)], tAsA[(None, e_s)], tma_bar_ptr=j_full, **e_cp)
                        cute.copy(tma_atom_sfa, e_gs[(None, e_s)], tAsSFA[(None, e_s)], tma_bar_ptr=j_full, **e_cp)
        if cutlass.const_expr(_HPDLW and _HPF):
            if warp_idx == self.tma_warp_id:
                pf_lane = cute.arch.lane_idx()
                pf_gx, pf_gy, pf_gz = cute.arch.grid_dim()
                for pf_j in cutlass.range_constexpr(_PPFN):
                    pf_b = cute.arch.block_idx()[2] + Int32(pf_j) * pf_gz
                    if pf_b < Int32(36 * 32 // _PCL):
                        pf_m = pf_b % Int32(32 // _PCL)
                        pf_l = pf_b // Int32(32 // _PCL)
                        if cutlass.const_expr(_PCL > 1):
                            pf_m = pf_m * _PCL + cute.arch.block_idx()[0] % _PCL
                        pf_w = j_ws[16] + (Int64(pf_l) * Int64(4096) + Int64(pf_m) * Int64(128)) * Int64(2048) + Int64(pf_lane) * Int64(8192)
                        if pf_j == 0 or pf_lane < _PPFL2:
                            l_prefetch_l2(pf_w, 8192)
                        if pf_lane < 2:
                            l_prefetch_l2(j_ws[17] + (Int64(pf_l) * Int64(32) + Int64(pf_m)) * Int64(32768) + Int64(pf_lane) * Int64(16384), 16384)
        if cutlass.const_expr(_HPDLW):
            if cutlass.const_expr(_PF):
                # lane P: TMA + MMA warps wait here; epilogue warps wait after zero-fill, gate warp after its bank reset
                if warp_idx == self.mma_warp_id or warp_idx == self.tma_warp_id:
                    if cutlass.const_expr(_PTR):
                        if warp_idx == self.tma_warp_id and cute.arch.lane_idx() == 0:
                            pgx4, pgy4, pgz4 = cute.arch.grid_dim()
                            pbx4, pby4, pbz4 = cute.arch.block_idx()
                            p_gates[2 * _PNG + 8 + 8 * ((pbz4 * pgy4 + pby4) * pgx4 + pbx4) + 5] = j_gtimer()
                    cute.arch.griddepcontrol_wait()
                    if cutlass.const_expr(_PTR):
                        if warp_idx == self.tma_warp_id and cute.arch.lane_idx() == 0:
                            pgx4, pgy4, pgz4 = cute.arch.grid_dim()
                            pbx4, pby4, pbz4 = cute.arch.block_idx()
                            p_gates[2 * _PNG + 8 + 8 * ((pbz4 * pgy4 + pby4) * pgx4 + pbx4) + 3] = j_gtimer()
            else:
                cute.arch.griddepcontrol_wait()  # lane L/N: routing (counts/SFB/xg) must be complete

        if cutlass.const_expr(_PF and _PG):
            if warp_idx == self.p_gate_warp_id and cutlass.const_expr(_PG != 4):
                g_lok = cutlass.Boolean(True)
                if cutlass.const_expr(_PGL == 1):
                    g_lok = cute.arch.lane_idx() == 0
                if g_lok:  # lane P: P_GL=32 keeps the gate warp warp-uniform (all lanes run the same walk)
                    g_base = Int64(p_gates.iterator.toint())
                    ggx, ggy, ggz = cute.arch.grid_dim()
                    gbx, gby, gbz = cute.arch.block_idx()
                    g_G = ggx * ggy * ggz
                    # launch epoch (one atomic per CTA per launch) -> gate bank parity; clear the other bank for the next launch
                    g_ep = p_atom_add(g_base + Int64(4 * (2 * _PNG + 2)), Int32(1))
                    g_bank = (g_ep // (g_G * _PGL)) % 2
                    g_i = (gbz * ggy + gby) * ggx + gbx
                    g_i0 = g_i
                    while g_i < _PNG:
                        p_gates[(1 - g_bank) * _PNG + g_i] = Int32(0)
                        g_i += g_G
                    if cutlass.const_expr(_HPDLW):
                        cute.arch.griddepcontrol_wait()
                    if cutlass.const_expr(_PTR):
                        p_gates[2 * _PNG + 8 + 8 * g_i0] = j_gtimer()
                    if cutlass.const_expr(_PG == 14):
                        for g_s in cutlass.range(40, unroll=1):
                            p_nanosleep(1000)
                    g_sched = self.j_mk_sched(False, False, tile_sched_params, None)
                    g_work = g_sched.initial_work_tile_info()
                    g_q = Int32(0)
                    g_dead = Int32(777)
                    if cutlass.const_expr(_PG == 16):
                        g_dead = p_gates[2 * _PNG + 5]  # always 0 at runtime: walk code present but never executed
                    while g_work.is_valid_tile and g_dead == 777 and cutlass.const_expr(_PG != 5 and _PG != 13 and _PG != 15):
                        g_cur = g_work.tile_idx
                        if (g_cur[3] & 1) == 1 and cutlass.const_expr(_PG != 6):
                            g_addr = g_base + Int64(4) * Int64(g_bank * _PNG + g_cur[2] * 64 + g_cur[1])
                            if cutlass.const_expr(_PG < 7):
                                if cutlass.const_expr(_PRLX):
                                    g_v = p_unig(p_ld_relaxed(g_addr))
                                    while g_v < 32:
                                        p_nanosleep(_PSLEEP)
                                        g_v = p_unig(p_ld_relaxed(g_addr))
                                    j_fence_gpu()
                                else:
                                    g_v = p_unig(j_ld_acquire(g_addr))
                                    while g_v < 32:
                                        p_nanosleep(_PSLEEP)
                                        g_v = p_unig(j_ld_acquire(g_addr))
                                p_fence_proxy_async_global()
                            g_q += 1
                            if cutlass.const_expr(_PG == 8):
                                p_gates[2 * _PNG + 4] = g_q
                            elif cutlass.const_expr(_PG == 9):
                                p_st_vol_shared(p_flag_addr + 8, g_q)
                            elif cutlass.const_expr(_PG == 10):
                                p_st_vol_shared(Int32(sW.iterator.toint()) + 200, g_q)
                            elif cutlass.const_expr(_PG == 11):
                                p_st_vol_global(g_base + Int64(4 * (2 * _PNG + 4)), g_q)
                            elif cutlass.const_expr(_PG == 12 or _PG == 14 or _PG == 16):
                                p_nanosleep(0)
                            elif cutlass.const_expr(_PRLX == 2):
                                p_st_vol_shared(p_flag_addr, g_q)
                            else:
                                p_st_rel_shared(p_flag_addr, g_q)
                        g_sched.advance_to_next_work()
                        g_work, _ = g_sched.get_current_work()
                    if cutlass.const_expr(_PTR):
                        p_gates[2 * _PNG + 8 + 8 * g_i0 + 1] = j_gtimer()

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            if cutlass.const_expr(_HS == 3):
                tile_sched = UnitScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()
                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage // _PK
                )
                while work_tile.is_valid_tile:
                    cur = work_tile.tile_idx
                    m_t = cur[0]
                    l_t = cur[2]
                    for kb in cutlass.range(k_block_cnt, unroll=1):
                        ab_pipeline.producer_acquire(ab_producer_state)
                        bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                        cute.copy(tma_atom_a, tAgA[(None, m_t, kb, l_t)], tAsA[(None, ab_producer_state.index)],
                                  tma_bar_ptr=bar, mcast_mask=a_full_mcast_mask)
                        cute.copy(tma_atom_sfa, tAgSFA[(None, m_t, kb, l_t)], tAsSFA[(None, ab_producer_state.index)],
                                  tma_bar_ptr=bar, mcast_mask=sfa_full_mcast_mask)
                        ab_producer_state.advance()
                    tile_sched.advance_to_next_unit()
                    work_tile, _ = tile_sched.get_current_work()
                ab_pipeline.producer_tail(ab_producer_state)
            elif cutlass.const_expr(_HS == 2):
                # Lane H weight-once mainloop: unit = (m-tile, expert); per k-block load the weight
                # (A+SFA) once and the token subtiles j = 0..n_sub-1 into consecutive stages
                # (stage j>0 carries only B+SFB; its A slot is unused).
                tile_sched = self.j_mk_sched(True, False, tile_sched_params, j_ctx)
                work_tile = tile_sched.initial_work_tile_info()
                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage // _PK
                )
                h_tile_n = self.cta_tile_shape_mnk[1]
                while work_tile.is_valid_tile:
                    cur = work_tile.tile_idx
                    m_t = cur[0]
                    l_t = cur[2]
                    n_sub = (tile_sched_params.masked_m[l_t] + (h_tile_n - 1)) // h_tile_n
                    for kb in cutlass.range(k_block_cnt, unroll=1):
                        for j in cutlass.range(n_sub, unroll=1):
                            tx = Int32(self.h_b_tx)
                            if j == 0:
                                tx = Int32(self.num_tma_load_bytes)
                            j_nb = Int32(h_tile_n // 32)
                            if cutlass.const_expr(_JV):
                                j_rem = tile_sched_params.masked_m[l_t] - j * h_tile_n
                                if j_rem > h_tile_n:
                                    j_rem = Int32(h_tile_n)
                                j_nb = (j_rem + 31) // 32
                                tx = Int32(self.j_sfb_tx) + j_nb * Int32(self.j_b32_bytes)
                                if j == 0:
                                    tx = tx + Int32(self.j_base_tx - self.j_sfb_tx)
                            ab_pipeline.producer_acquire(ab_producer_state, expected_tx=tx)
                            bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                            if j == 0:
                                cute.copy(
                                    tma_atom_a,
                                    tAgA[(None, m_t, kb, l_t)],
                                    tAsA[(None, ab_producer_state.index)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=a_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_sfa,
                                    tAgSFA[(None, m_t, kb, l_t)],
                                    tAsSFA[(None, ab_producer_state.index)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfa_full_mcast_mask,
                                )
                            sfb_n = j
                            if cutlass.const_expr(h_tile_n < 128):
                                sfb_n = j // (128 // h_tile_n)
                            if cutlass.const_expr(_JV):
                                for jb in cutlass.range(j_nb, unroll=1):
                                    cute.copy(
                                        j_tma_atom_b32,
                                        tBgB32[(None, j * (h_tile_n // 32) + jb, kb, l_t)],
                                        tBsB32[(None, ab_producer_state.index * (h_tile_n // 32) + jb)],
                                        tma_bar_ptr=bar,
                                        mcast_mask=b_full_mcast_mask,
                                    )
                            else:
                                cute.copy(
                                    tma_atom_b,
                                    tBgB[(None, j, kb, l_t)],
                                    tBsB[(None, ab_producer_state.index)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB[(None, sfb_n, kb, l_t)],
                                tBsSFB[(None, ab_producer_state.index)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                            ab_producer_state.advance()
                    tile_sched.advance_to_next_unit()
                    work_tile, _ = tile_sched.get_current_work()
                ab_pipeline.producer_tail(ab_producer_state)
            else:
                #
                # Persistent tile scheduling loop
                #
                tile_sched = self.j_mk_sched(True, False, tile_sched_params, j_ctx)
                work_tile = tile_sched.initial_work_tile_info()

                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage // _PK
                )
                j_t = Int32(0)
                p_q = Int32(0)
                p_fi = Int32(1)
                p_cp = {}
                if cutlass.const_expr(_PF and _PEVF):
                    p_cp = {"cache_policy": p_mkpol("evict_first")}  # lane P: weights are read once -> evict first
                p_la_v = Int32(32)
                p_gb = Int64(0)
                p_pbank = Int32(0)
                p_las = self.j_mk_sched(False, False, tile_sched_params, None)
                if cutlass.const_expr(_PF and (_PG == 13 or _PG == 15)):
                    # lane P producer-side gating: one warp atomic (32 lanes -> +32 per CTA) gives the launch epoch/bank;
                    # a look-ahead scheduler issues the NEXT tile's relaxed gate load one tile early (latency hidden).
                    p_gb = Int64(p_gates.iterator.toint())
                    ppx, ppy, ppz = cute.arch.grid_dim()
                    p_pe = p_atom_add(p_gb + Int64(4 * (2 * _PNG + 3)), Int32(1))
                    p_pbank = (p_pe // (ppx * ppy * ppz * 32)) % 2
                    p_las.advance_to_next_work()
                if cutlass.const_expr(_PF and _PTR):
                    if cute.arch.lane_idx() == 0:
                        pgx5, pgy5, pgz5 = cute.arch.grid_dim()
                        pbx5, pby5, pbz5 = cute.arch.block_idx()
                        p_gates[2 * _PNG + 8 + 8 * ((pbz5 * pgy5 + pby5) * pgx5 + pbx5) + 4] = j_gtimer()
                while work_tile.is_valid_tile:
                    if j_t < 16:
                        self.j_trace(j_order, bidz, 1 + j_t)
                    j_t += 1
                    # Get tile coord from tile scheduler
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_mnl = (
                        cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                        cur_tile_coord[1],
                        cur_tile_coord[2],
                    )

                    #
                    # Slice to per mma tile index
                    #
                    # ((atom_v, rest_v), RestK)
                    tAgA_slice = tAgA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    # ((atom_v, rest_v), RestK)
                    tBgB_slice = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]

                    # ((atom_v, rest_v), RestK)
                    tAgSFA_slice = tAgSFA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    # ((atom_v, rest_v), RestK)
                    slice_n = mma_tile_coord_mnl[1]
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] < 128):
                        slice_n = mma_tile_coord_mnl[1] // (128 // self.cta_tile_shape_mnk[1])
                    tBgSFB_slice = tBgSFB[
                        (None, slice_n, None, mma_tile_coord_mnl[2])
                    ]

                    j_nv = Int32(self.cta_tile_shape_mnk[1] // 32)
                    j_tx = Int32(self.num_tma_load_bytes)
                    j_kb0 = Int32(0)
                    j_nk = k_block_cnt
                    if cutlass.const_expr(_PF):
                        j_nk = Int32(k_block_cnt)
                    if cutlass.const_expr(_JS):
                        j_w1 = cur_tile_coord[3]
                        j_kb0 = j_w1 & 31
                        j_nk = ((j_w1 >> 5) & 31) - j_kb0
                    p_ph = Int32(0)
                    p_cur_v = p_la_v
                    if cutlass.const_expr(_PF and (_PG == 13 or _PG == 15)):
                        p_law, _ = p_las.get_current_work()
                        p_la_v = Int32(32)
                        if p_law.is_valid_tile:
                            p_lac = p_law.tile_idx
                            if p_lac[3] == 1:
                                p_la_v = p_uni(p_ld_relaxed(p_gb + Int64(4) * Int64(p_pbank * _PNG + p_lac[2] * 64 + p_lac[1])))
                        p_las.advance_to_next_work()
                    p_em = Int32(0)
                    if cutlass.const_expr(_PF):
                        p_ph = cur_tile_coord[3] & 1
                        if cutlass.const_expr(_PEA > 0):
                            if p_fi == 1:
                                p_fi = Int32(0)
                                e_c2 = cute.arch.block_idx()[2]
                                if e_c2 // Int32(32) < Int32(_PEAX):
                                    cute.arch.mbarrier_wait(j_full, 0)  # early A/SFA landed (also makes a mismatch overwrite safe)
                                    if p_ph == 0 and cur_tile_coord[2] == e_c2 // Int32(32) and mma_tile_coord_mnl[0] == e_c2 % Int32(32):
                                        p_em = Int32(1)
                        tAgA2_slice = tAgA2[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                        tAgSFA2_slice = tAgSFA2[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                        tBgSFB2_slice = tBgSFB2[(None, slice_n, None, mma_tile_coord_mnl[2])]
                        if p_ph == 1:
                            j_nk = Int32(k_block_cnt2)
                            if cutlass.const_expr(_PSPL == 2):
                                j_nk = Int32(k_block_cnt2 // 2)
                                j_kb0 = (cur_tile_coord[3] >> 2) * Int32(k_block_cnt2 // 2)
                            if cutlass.const_expr(_PG == 1 or _PG == 3):
                                if cutlass.const_expr(_PRLX):
                                    p_r = p_uni(p_ld_vol_shared(p_flag_addr))
                                    while p_r <= p_q:
                                        p_r = p_uni(p_ld_vol_shared(p_flag_addr))
                                else:
                                    p_r = p_uni(p_ld_acq_shared(p_flag_addr))
                                    while p_r <= p_q:
                                        p_r = p_uni(p_ld_acq_shared(p_flag_addr))
                                p_q = p_q + 1
                            if cutlass.const_expr(_PG == 13 or _PG == 15):
                                if p_cur_v < 32:
                                    p_ga = p_gb + Int64(4) * Int64(p_pbank * _PNG + cur_tile_coord[2] * 64 + cur_tile_coord[1])
                                    p_cur_v = p_uni(p_ld_relaxed(p_ga))
                                    while p_cur_v < 32:
                                        p_nanosleep(_PSLEEP)
                                        p_cur_v = p_uni(p_ld_relaxed(p_ga))
                                if cutlass.const_expr(_PG == 13):
                                    j_fence_gpu()
                                p_fence_proxy_async_global()
                    if cutlass.const_expr(_JV):
                        j_rem = tile_sched_params.masked_m[mma_tile_coord_mnl[2]] - mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1]
                        if j_rem > self.cta_tile_shape_mnk[1]:
                            j_rem = Int32(self.cta_tile_shape_mnk[1])
                        j_nv = (j_rem + 31) // 32
                        if cutlass.const_expr(_JX == 2):
                            j_nv = Int32(self.cta_tile_shape_mnk[1] // 32)
                        j_tx = Int32(self.j_base_tx) + j_nv * Int32(self.j_b32_bytes)
                    if cutlass.const_expr(_PNOB):
                        j_nv = Int32(0)
                        j_tx = Int32(self.j_base_tx)
                    p_ska = Int32(0)
                    if cutlass.const_expr(_PNOLD):
                        j_tx = Int32(0)
                    if cutlass.const_expr(_PASK):
                        # round 6 bound (timing only, output wrong): odd token tiles reuse the even tile's A -> no A/SFA load
                        if (mma_tile_coord_mnl[1] & 1) == 1:
                            p_ska = Int32(1)
                            j_tx = j_tx - Int32(self.p_aw_tx)
                    if cutlass.const_expr(_PK > 1):
                        j_nk = j_nk // _PK
                        j_tx = j_tx * _PK
                    # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt
                    ab_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < j_nk:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )
                    #
                    # Tma load loop
                    #
                    for k_block in cutlass.range(0, j_nk, 1, unroll=1):  # noqa: B007
                        # Conditionally wait for AB buffer empty
                        if cutlass.const_expr(_JV):
                            p_etx = j_tx
                            if cutlass.const_expr(_PEA > 0):
                                if p_em == 1 and ab_producer_state.count < _PEA:
                                    p_etx = j_tx - Int32(self.p_aw_tx)
                            ab_pipeline.producer_acquire(
                                ab_producer_state, peek_ab_empty_status, expected_tx=p_etx
                            )
                        else:
                            ab_pipeline.producer_acquire(
                                ab_producer_state, peek_ab_empty_status
                            )

                        # TMA load A/B/SFA/SFB
                        if cutlass.const_expr(_PF):
                            p_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                            for p_sub in cutlass.range_constexpr(0 if _PNOLD else _PK):
                                p_kb = ab_producer_state.count * _PK + p_sub + j_kb0
                                p_n32 = mma_tile_coord_mnl[1] * (self.cta_tile_shape_mnk[1] // 32)
                                p_si = ab_producer_state.index * _PK + p_sub
                                p_skw = cutlass.Boolean(False)
                                if cutlass.const_expr(_PEA > 0):
                                    p_skw = p_em == 1 and ab_producer_state.count < _PEA
                                if cutlass.const_expr(_PASK):
                                    p_skw = p_ska == 1
                                if p_ph == 1 and p_skw:
                                    for jb in cutlass.range(j_nv, unroll=1):
                                        cute.copy(tma_atom_b32_2, tBgB32_2[(None, p_n32 + jb, p_kb, mma_tile_coord_mnl[2])],
                                                  tBsB32_2[(None, p_si * (self.cta_tile_shape_mnk[1] // 32) + jb)], tma_bar_ptr=p_bar, mcast_mask=b_full_mcast_mask)
                                    cute.copy(tma_atom_sfb2, tBgSFB2_slice[(None, p_kb)], tBsSFB2[(None, p_si)], tma_bar_ptr=p_bar, mcast_mask=sfb_full_mcast_mask)
                                elif p_ph == 0 and p_skw:
                                    for jb in cutlass.range(j_nv, unroll=1):
                                        cute.copy(j_tma_atom_b32, tBgB32[(None, p_n32 + jb, p_kb, mma_tile_coord_mnl[2])],
                                                  tBsB32[(None, p_si * (self.cta_tile_shape_mnk[1] // 32) + jb)], tma_bar_ptr=p_bar, mcast_mask=b_full_mcast_mask)
                                    cute.copy(tma_atom_sfb, tBgSFB_slice[(None, p_kb)], tBsSFB[(None, p_si)], tma_bar_ptr=p_bar, mcast_mask=sfb_full_mcast_mask)
                                elif p_ph == 0:
                                    cute.copy(tma_atom_a, tAgA_slice[(None, p_kb)], tAsA[(None, p_si)], tma_bar_ptr=p_bar, mcast_mask=a_full_mcast_mask, **p_cp)
                                    for jb in cutlass.range(j_nv, unroll=1):
                                        cute.copy(j_tma_atom_b32, tBgB32[(None, p_n32 + jb, p_kb, mma_tile_coord_mnl[2])],
                                                  tBsB32[(None, p_si * (self.cta_tile_shape_mnk[1] // 32) + jb)], tma_bar_ptr=p_bar, mcast_mask=b_full_mcast_mask)
                                    cute.copy(tma_atom_sfa, tAgSFA_slice[(None, p_kb)], tAsSFA[(None, p_si)], tma_bar_ptr=p_bar, mcast_mask=sfa_full_mcast_mask, **p_cp)
                                    cute.copy(tma_atom_sfb, tBgSFB_slice[(None, p_kb)], tBsSFB[(None, p_si)], tma_bar_ptr=p_bar, mcast_mask=sfb_full_mcast_mask)
                                else:
                                    cute.copy(tma_atom_a2, tAgA2_slice[(None, p_kb)], tAsA2[(None, p_si)], tma_bar_ptr=p_bar, mcast_mask=a_full_mcast_mask, **p_cp)
                                    for jb in cutlass.range(j_nv, unroll=1):
                                        cute.copy(tma_atom_b32_2, tBgB32_2[(None, p_n32 + jb, p_kb, mma_tile_coord_mnl[2])],
                                                  tBsB32_2[(None, p_si * (self.cta_tile_shape_mnk[1] // 32) + jb)], tma_bar_ptr=p_bar, mcast_mask=b_full_mcast_mask)
                                    cute.copy(tma_atom_sfa2, tAgSFA2_slice[(None, p_kb)], tAsSFA2[(None, p_si)], tma_bar_ptr=p_bar, mcast_mask=sfa_full_mcast_mask, **p_cp)
                                    cute.copy(tma_atom_sfb2, tBgSFB2_slice[(None, p_kb)], tBsSFB2[(None, p_si)], tma_bar_ptr=p_bar, mcast_mask=sfb_full_mcast_mask)
                        else:
                          cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, ab_producer_state.count + j_kb0)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=a_full_mcast_mask,
                        )
                        if cutlass.const_expr(_JV and not _PF):
                            for jb in cutlass.range(j_nv, unroll=1):
                                cute.copy(
                                    j_tma_atom_b32,
                                    tBgB32[(None, mma_tile_coord_mnl[1] * (self.cta_tile_shape_mnk[1] // 32) + jb, ab_producer_state.count + j_kb0, mma_tile_coord_mnl[2])],
                                    tBsB32[(None, ab_producer_state.index * (self.cta_tile_shape_mnk[1] // 32) + jb)],
                                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                                    mcast_mask=b_full_mcast_mask,
                                )
                        elif cutlass.const_expr(_GNOB == 0 and not _PF):
                          cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, ab_producer_state.count + j_kb0)],
                            tBsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=b_full_mcast_mask,
                        )
                        if cutlass.const_expr(not _PF):
                          cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, ab_producer_state.count + j_kb0)],
                            tAsSFA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=sfa_full_mcast_mask,
                          )
                        if cutlass.const_expr(_GNOB == 0 and not _PF):
                          cute.copy(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, ab_producer_state.count + j_kb0)],
                            tBsSFB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=sfb_full_mcast_mask,
                        )

                        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt + k_block + 1
                        ab_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if ab_producer_state.count < j_nk:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                ab_producer_state
                            )

                    #
                    # Advance to next tile
                    #
                    tile_sched.advance_to_next_work()
                    work_tile, _ = tile_sched.get_current_work()

                #
                # Wait A/B buffer empty
                #
                ab_pipeline.producer_tail(ab_producer_state)

        if cutlass.const_expr(_HS == 3):
            if warp_idx == self.tmab_warp_id:
                tile_sched = UnitScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()
                b_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_b_stage
                )
                h_tile_n = self.cta_tile_shape_mnk[1]
                while work_tile.is_valid_tile:
                    cur = work_tile.tile_idx
                    l_t = cur[2]
                    n_sub = (tile_sched_params.masked_m[l_t] + (h_tile_n - 1)) // h_tile_n
                    for kb in cutlass.range(k_block_cnt, unroll=1):
                        for j in cutlass.range(n_sub, unroll=1):
                            b_pipeline.producer_acquire(b_producer_state)
                            bar = b_pipeline.producer_get_barrier(b_producer_state)
                            sfb_n = j
                            if cutlass.const_expr(h_tile_n < 128):
                                sfb_n = j // (128 // h_tile_n)
                            cute.copy(tma_atom_b, tBgB[(None, j, kb, l_t)], tBsB[(None, b_producer_state.index)],
                                      tma_bar_ptr=bar, mcast_mask=b_full_mcast_mask)
                            cute.copy(tma_atom_sfb, tBgSFB[(None, sfb_n, kb, l_t)], tBsSFB[(None, b_producer_state.index)],
                                      tma_bar_ptr=bar, mcast_mask=sfb_full_mcast_mask)
                            b_producer_state.advance()
                    tile_sched.advance_to_next_unit()
                    work_tile, _ = tile_sched.get_current_work()
                b_pipeline.producer_tail(b_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem_ptr_read_threads = self.threads_per_warp * len(
                (self.mma_warp_id, *self.epilog_warp_id)
            )
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            # Make accumulator tmem tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # Partition for S2T copy of SFA/SFB
            #
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            if cutlass.const_expr(_HS >= 2):
                tile_sched = self.j_mk_sched(False, False, tile_sched_params, j_ctx)
                work_tile = tile_sched.initial_work_tile_info()
                ab_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage // _PK
                )
                acc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_acc_stage
                )
                b_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_b_stage
                )
                h_tile_n = self.cta_tile_shape_mnk[1]
                sfb_col0 = tcgen05.find_tmem_tensor_col_offset(tCtAcc_base) + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                print('[H] tmem acc cols', tcgen05.find_tmem_tensor_col_offset(tCtAcc_base), 'sfa cols', tcgen05.find_tmem_tensor_col_offset(tCtSFA), 'sfb cols', tcgen05.find_tmem_tensor_col_offset(tCtSFB), flush=True)
                num_kphases = cute.size(tCrA, mode=[2])
                while work_tile.is_valid_tile:
                    cur = work_tile.tile_idx
                    l_t = cur[2]
                    n_sub = (tile_sched_params.masked_m[l_t] + (h_tile_n - 1)) // h_tile_n
                    acc_base = acc_producer_state.index
                    acc_commit_state = acc_producer_state.clone()
                    for j in cutlass.range(n_sub, unroll=1):
                        acc_pipeline.producer_acquire(acc_producer_state)
                        acc_producer_state.advance()
                    for kb in cutlass.range(k_block_cnt, unroll=1):
                        a_state = ab_consumer_state.clone()
                        if cutlass.const_expr(_HS == 3):
                            ab_pipeline.consumer_wait(ab_consumer_state)
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[(None, None, None, None, ab_consumer_state.index)],
                                tCtSFA_compact_s2t,
                            )
                            ab_consumer_state.advance()
                        for j in cutlass.range(n_sub, unroll=1):
                            if cutlass.const_expr(_HS == 3):
                                bst = b_consumer_state
                                b_pipeline.consumer_wait(b_consumer_state)
                            else:
                                bst = ab_consumer_state
                                ab_pipeline.consumer_wait(ab_consumer_state)
                                if j == 0:
                                    cute.copy(
                                        tiled_copy_s2t_sfa,
                                        tCsSFA_compact_s2t[(None, None, None, None, ab_consumer_state.index)],
                                        tCtSFA_compact_s2t,
                                    )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[(None, None, None, None, bst.index)],
                                tCtSFB_compact_s2t,
                            )
                            acc_idx = (acc_base + j) % self.num_acc_stage
                            tCtAcc = tCtAcc_base[(None, None, None, acc_idx)]
                            tCtSFB_mma = tCtSFB
                            if cutlass.const_expr(h_tile_n < 128):
                                tCtSFB_mma = cute.make_tensor(
                                    cute.recast_ptr(
                                        acc_tmem_ptr + sfb_col0 + (j % (128 // h_tile_n)) * (4 * h_tile_n // 128),
                                        dtype=self.sf_dtype,
                                    ),
                                    tCtSFB_layout,
                                )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, kb != 0)
                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kphase_idx)].iterator)
                                tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[(None, None, kphase_idx)].iterator)
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[(None, None, kphase_idx, a_state.index)],
                                    tCrB[(None, None, kphase_idx, bst.index)],
                                    tCtAcc,
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            if cutlass.const_expr(_HS == 3):
                                b_pipeline.consumer_release(b_consumer_state)
                                b_consumer_state.advance()
                            else:
                                if j != 0:
                                    ab_pipeline.consumer_release(ab_consumer_state)
                                ab_consumer_state.advance()
                        ab_pipeline.consumer_release(a_state)
                    for j in cutlass.range(n_sub, unroll=1):
                        acc_pipeline.producer_commit(acc_commit_state)
                        acc_commit_state.advance()
                    tile_sched.advance_to_next_unit()
                    work_tile, _ = tile_sched.get_current_work()
                acc_pipeline.producer_tail(acc_producer_state)
            else:
                #
                # Persistent tile scheduling loop
                #
                tile_sched = self.j_mk_sched(False, False, tile_sched_params, j_ctx)
                work_tile = tile_sched.initial_work_tile_info()

                ab_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage // _PK
                )
                acc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_acc_stage
                )

                while work_tile.is_valid_tile:
                    # Get tile coord from tile scheduler
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_mnl = (
                        cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                        cur_tile_coord[1],
                        cur_tile_coord[2],
                    )

                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)
                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                    j_nk = k_block_cnt
                    if cutlass.const_expr(_PF):
                        j_nk = Int32(k_block_cnt)
                        if (cur_tile_coord[3] & 1) == 1:
                            j_nk = Int32(k_block_cnt2)
                            if cutlass.const_expr(_PSPL == 2):
                                j_nk = Int32(k_block_cnt2 // 2)
                        if cutlass.const_expr(_PK > 1):
                            j_nk = j_nk // _PK
                    if cutlass.const_expr(_JS):
                        j_w1 = cur_tile_coord[3]
                        j_nk = ((j_w1 >> 5) & 31) - (j_w1 & 31)
                    # Peek (try_wait) AB buffer full for k_block = 0
                    ab_consumer_state.reset_count()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < j_nk and is_leader_cta:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state
                        )

                    #
                    # Wait for accumulator buffer empty
                    #
                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state)

                    #
                    # Reset the ACCUMULATE field for each tile
                    #
                    tCtSFB_mma = tCtSFB
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] < 128):
                        sfb_off = cutlass.Int32((mma_tile_coord_mnl[1] % (128 // self.cta_tile_shape_mnk[1])) * (4 * self.cta_tile_shape_mnk[1] // 128))
                        tCtSFB_mma = cute.make_tensor(
                            cute.recast_ptr(
                                acc_tmem_ptr
                                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                                + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                                + sfb_off,
                                dtype=self.sf_dtype,
                            ),
                            tCtSFB_layout,
                        )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    if cutlass.const_expr(_JV):
                        j_rem = tile_sched_params.masked_m[mma_tile_coord_mnl[2]] - mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1]
                        if j_rem > self.cta_tile_shape_mnk[1]:
                            j_rem = Int32(self.cta_tile_shape_mnk[1])
                        j_nv = (j_rem + 31) // 32
                        if cutlass.const_expr(_JX == 1):
                            j_nv = Int32(4)
                        j_acc32 = cute.make_tensor(tCtAcc.iterator, j_acc32_layout)
                        j_acc64 = cute.make_tensor(tCtAcc.iterator, j_acc64_layout)

                    #
                    # Mma mainloop
                    #
                    if cutlass.const_expr(_JS or _PF):
                        for k_block in cutlass.range(j_nk, unroll=1):  # noqa: B007
                            if is_leader_cta:
                                # Conditionally wait for AB buffer full
                                ab_pipeline.consumer_wait(
                                    ab_consumer_state, peek_ab_full_status
                                )

                                for m_sub in cutlass.range_constexpr(_PK):
                                    m_si = ab_consumer_state.index * _PK + m_sub
                                    #  Copy SFA/SFB from smem to tmem
                                    s2t_stage_coord = (
                                        None,
                                        None,
                                        None,
                                        None,
                                        m_si,
                                    )
                                    tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                                    tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                                    if cutlass.const_expr(_GNOS2T == 0):
                                        cute.copy(
                                            tiled_copy_s2t_sfa,
                                            tCsSFA_compact_s2t_staged,
                                            tCtSFA_compact_s2t,
                                        )
                                        cute.copy(
                                            tiled_copy_s2t_sfb,
                                            tCsSFB_compact_s2t_staged,
                                            tCtSFB_compact_s2t,
                                        )

                                    # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                                    num_kphases = cute.size(tCrA, mode=[2])
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (
                                            None,
                                            None,
                                            kphase_idx,
                                            m_si,
                                        )

                                        # Set SFA/SFB tensor to tiled_mma
                                        sf_kphase_coord = (None, None, kphase_idx)
                                        tiled_mma.set(
                                            tcgen05.Field.SFA,
                                            tCtSFA[sf_kphase_coord].iterator,
                                        )
                                        tiled_mma.set(
                                            tcgen05.Field.SFB,
                                            tCtSFB_mma[sf_kphase_coord].iterator,
                                        )

                                        if cutlass.const_expr(_JV and not _PF):
                                            j_acc_on = (kphase_idx + k_block) != 0
                                            tiled_mma.set(tcgen05.Field.ACCUMULATE, j_acc_on)
                                            j_tm32.set(tcgen05.Field.ACCUMULATE, j_acc_on)
                                            j_tm32.set(tcgen05.Field.SFA, tCtSFA[sf_kphase_coord].iterator)
                                            j_tm32.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kphase_coord].iterator)
                                            j_tm64.set(tcgen05.Field.ACCUMULATE, j_acc_on)
                                            j_tm64.set(tcgen05.Field.SFA, tCtSFA[sf_kphase_coord].iterator)
                                            j_tm64.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kphase_coord].iterator)
                                            if j_nv == 1:
                                                cute.gemm(
                                                    j_tm32,
                                                    j_acc32,
                                                    tCrA[kphase_coord],
                                                    tCrB32[(None, None, kphase_idx, m_si * (self.cta_tile_shape_mnk[1] // 32))],
                                                    j_acc32,
                                                )
                                            elif j_nv == 2:
                                                cute.gemm(
                                                    j_tm64,
                                                    j_acc64,
                                                    tCrA[kphase_coord],
                                                    tCrB64[(None, None, kphase_idx, m_si * (self.cta_tile_shape_mnk[1] // 64))],
                                                    j_acc64,
                                                )
                                            else:
                                                cute.gemm(
                                                    tiled_mma,
                                                    tCtAcc,
                                                    tCrA[kphase_coord],
                                                    tCrB[kphase_coord],
                                                    tCtAcc,
                                                )
                                        elif cutlass.const_expr(_GNOMMA == 0):
                                            cute.gemm(
                                                tiled_mma,
                                                tCtAcc,
                                                tCrA[kphase_coord],
                                                tCrB[kphase_coord],
                                                tCtAcc,
                                            )

                                        # Enable accumulate on tCtAcc after first kphase
                                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                                # Async arrive AB buffer empty
                                ab_pipeline.consumer_release(ab_consumer_state)

                            # Peek (try_wait) AB buffer full for k_block = k_block + 1
                            ab_consumer_state.advance()
                            peek_ab_full_status = cutlass.Boolean(1)
                            if ab_consumer_state.count < j_nk:
                                if is_leader_cta:
                                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                        ab_consumer_state
                                    )
                    else:
                        for k_block in cutlass.range_constexpr(k_block_cnt):  # noqa: B007
                            if is_leader_cta:
                                # Conditionally wait for AB buffer full
                                ab_pipeline.consumer_wait(
                                    ab_consumer_state, peek_ab_full_status
                                )

                                #  Copy SFA/SFB from smem to tmem
                                s2t_stage_coord = (
                                    None,
                                    None,
                                    None,
                                    None,
                                    ab_consumer_state.index,
                                )
                                tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                                tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                                if cutlass.const_expr(_GNOS2T == 0):
                                    cute.copy(
                                        tiled_copy_s2t_sfa,
                                        tCsSFA_compact_s2t_staged,
                                        tCtSFA_compact_s2t,
                                    )
                                    cute.copy(
                                        tiled_copy_s2t_sfb,
                                        tCsSFB_compact_s2t_staged,
                                        tCtSFB_compact_s2t,
                                    )

                                # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                                num_kphases = cute.size(tCrA, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (
                                        None,
                                        None,
                                        kphase_idx,
                                        ab_consumer_state.index,
                                    )

                                    # Set SFA/SFB tensor to tiled_mma
                                    sf_kphase_coord = (None, None, kphase_idx)
                                    tiled_mma.set(
                                        tcgen05.Field.SFA,
                                        tCtSFA[sf_kphase_coord].iterator,
                                    )
                                    tiled_mma.set(
                                        tcgen05.Field.SFB,
                                        tCtSFB_mma[sf_kphase_coord].iterator,
                                    )

                                    if cutlass.const_expr(_JV and not _PF):
                                        j_acc_on = (kphase_idx + k_block) != 0
                                        tiled_mma.set(tcgen05.Field.ACCUMULATE, j_acc_on)
                                        j_tm32.set(tcgen05.Field.ACCUMULATE, j_acc_on)
                                        j_tm32.set(tcgen05.Field.SFA, tCtSFA[sf_kphase_coord].iterator)
                                        j_tm32.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kphase_coord].iterator)
                                        j_tm64.set(tcgen05.Field.ACCUMULATE, j_acc_on)
                                        j_tm64.set(tcgen05.Field.SFA, tCtSFA[sf_kphase_coord].iterator)
                                        j_tm64.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kphase_coord].iterator)
                                        if j_nv == 1:
                                            cute.gemm(
                                                j_tm32,
                                                j_acc32,
                                                tCrA[kphase_coord],
                                                tCrB32[(None, None, kphase_idx, ab_consumer_state.index * (self.cta_tile_shape_mnk[1] // 32))],
                                                j_acc32,
                                            )
                                        elif j_nv == 2:
                                            cute.gemm(
                                                j_tm64,
                                                j_acc64,
                                                tCrA[kphase_coord],
                                                tCrB64[(None, None, kphase_idx, ab_consumer_state.index * (self.cta_tile_shape_mnk[1] // 64))],
                                                j_acc64,
                                            )
                                        else:
                                            cute.gemm(
                                                tiled_mma,
                                                tCtAcc,
                                                tCrA[kphase_coord],
                                                tCrB[kphase_coord],
                                                tCtAcc,
                                            )
                                    elif cutlass.const_expr(_GNOMMA == 0):
                                        cute.gemm(
                                            tiled_mma,
                                            tCtAcc,
                                            tCrA[kphase_coord],
                                            tCrB[kphase_coord],
                                            tCtAcc,
                                        )

                                    # Enable accumulate on tCtAcc after first kphase
                                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                                # Async arrive AB buffer empty
                                ab_pipeline.consumer_release(ab_consumer_state)

                            # Peek (try_wait) AB buffer full for k_block = k_block + 1
                            ab_consumer_state.advance()
                            peek_ab_full_status = cutlass.Boolean(1)
                            if ab_consumer_state.count < j_nk:
                                if is_leader_cta:
                                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                        ab_consumer_state
                                    )

                    #
                    # Async arrive accumulator buffer full
                    #
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                    #
                    # Advance to next tile
                    #
                    tile_sched.advance_to_next_work()
                    work_tile, _ = tile_sched.get_current_work()

                #
                # Wait for accumulator buffer empty
                #
                acc_pipeline.producer_tail(acc_producer_state)
        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem_ptr_read_threads = self.threads_per_warp * len(
                (self.mma_warp_id, *self.epilog_warp_id)
            )
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            p_zs = Int32(0)
            if cutlass.const_expr(_PF):
                # lane P: zero out[T, H] (independent of routing) while the first tile's mainloop runs, then bump the
                # launch epoch counter (also selects the gate bank); FC2 epilogues wait once for all CTAs' zeroing.
                if cutlass.const_expr(_PZ and _PZT):
                    if cutlass.const_expr(_HPDLW):
                        cute.arch.griddepcontrol_wait()
                    zs_base = Int64(p_out.iterator.toint())
                    zs_T = p_gates[2 * _PNG + 1] // 512
                    zsx, zsy, zsz = cute.arch.grid_dim()
                    zbx, zby, zbz = cute.arch.block_idx()
                    zs_G = zsx * zsy * zsz
                    zs_r = (zbz * zsy + zby) * zsx + zbx
                    while zs_r < zs_T:
                        if p_gates[_PNL0 + zs_r] > 0:
                            for zq in cutlass.range_constexpr(4):
                                h_st_zero16(zs_base + Int64(zs_r) * Int64(8192) + Int64((zq * 128 + tidx) * 16))
                        zs_r += zs_G
                    cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=128)
                if cutlass.const_expr(_PZ and not _PZT and not _PZTL):
                    pz_base = Int64(p_out.iterator.toint())
                    pz_n = p_gates[2 * _PNG + 1]
                    pgx, pgy, pgz = cute.arch.grid_dim()
                    pbx, pby, pbz = cute.arch.block_idx()
                    pz_thr = pgx * pgy * pgz * 128
                    pzi = ((pbz * pgy + pby) * pgx + pbx) * 128 + tidx
                    while pzi < pz_n:
                        h_st_zero16(pz_base + Int64(pzi) * Int64(16))
                        pzi += pz_thr
                    cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=128)
                if tidx == 0:
                    j_fence_gpu()
                    p_zs = p_atom_add(Int64(p_gates.iterator.toint()) + Int64(4 * 2 * _PNG), Int32(1))
                    if cutlass.const_expr(not _PG):
                        p_atom_add(Int64(p_gates.iterator.toint()) + Int64(4 * (2 * _PNG + 2)), Int32(1))  # keep epochs in sync
                if cutlass.const_expr(_HPDLW and not (_PZ and _PZT)):
                    cute.arch.griddepcontrol_wait()

            if cutlass.const_expr(_HF):
                #
                # Lane H fused epilogue: alpha * acc -> SwiGLU (rows 0..63 up, 64..127 gate of each
                # 128-row weight tile) -> NVFP4 (block 16 along intermediate, e4m3 scale = gs*amax/6)
                # -> 8-byte stores of VALID token rows only + swizzled (32x4x4) scale bytes.
                #
                epi_threads = self.threads_per_warp * len(self.epilog_warp_id)
                epilog_threads = epi_threads
                tile_n = self.cta_tile_shape_mnk[1]
                tAcc_h = cute.flat_divide(
                    tCtAcc_base[((None, None), 0, 0, None)], (128, _HEPI_COLS)
                )
                t2r_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(getattr(tcgen05.copy.Repetition, 'x%d' % _HEPI_COLS)), self.acc_dtype
                )
                tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, tAcc_h[(None, None, 0, 0, 0)])
                thr_t2r = tiled_t2r.get_slice(tidx)
                tTR_tAcc_h = thr_t2r.partition_S(tAcc_h)
                tTR_cId = thr_t2r.partition_D(cute.make_identity_tensor((128, _HEPI_COLS)))
                tTR_r = cute.make_rmem_tensor(tTR_cId.shape, self.acc_dtype)
                gs = h_gs[0]
                inv_gs = cute.arch.rcp_approx(gs)
                rcp6 = cutlass.Float32(1.0 / 6.0)
                LOG2E = cutlass.Float32(1.4426950408889634)
                sf_rk = h_out.shape[2] // 4
                sf_rm = (h_out.shape[1] + 127) // 128
                q_c = tidx % _HEPI_COLS
                q_b = tidx // _HEPI_COLS

                tile_sched = self.j_mk_sched(False, True, tile_sched_params, j_ctx)
                j_t = Int32(0)
                if warp_idx == 0:
                    self.j_trace(j_order, bidz, 0)
                work_tile = tile_sched.initial_work_tile_info()
                acc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage
                )
                p_ndone = Int32(0)
                p_zready = Int32(0)
                p_zdone = Int32(0)
                p_gaddr = Int64(0)
                p_ztarget = Int32(0)
                p_bank = Int32(0)
                p_sbase = Int32(0)
                p_gbase = Int64(0)
                if cutlass.const_expr(_PF):
                    p_gaddr = Int64(p_gates.iterator.toint())
                    pgx2, pgy2, pgz2 = cute.arch.grid_dim()
                    p_G = pgx2 * pgy2 * pgz2
                    p_ztarget = (p_zs // p_G + 1) * p_G
                    p_bank = (p_zs // p_G) % 2
                    if cutlass.const_expr(_PZT):
                        pcx, pcy, pcz = cute.arch.block_idx()
                        if tidx == 0 and pcz == 0 and pcx == 0 and pcy == 0:
                            p_gates[2 * _PNG + 6 + (1 - p_bank)] = Int32(0)
                    p_sbase = Int32(sEpi2.iterator.toint())
                    p_gbase = Int64(p_out.iterator.toint())
                if cutlass.const_expr(_HZB > 0 and _HZPOS == 0):
                    # Lane I: zero the FC2 finalize output, which the caller placed right after the SF buffer
                    # (same allocation), while the epilogue warps wait for the first accumulator.
                    z_base = Int64(h_sf.iterator.toint()) + Int64(cute.cosize(h_sf.layout))
                    gdx, gdy, gdz = cute.arch.grid_dim()
                    bx, by, bz = cute.arch.block_idx()
                    nthr = gdx * gdy * gdz * epi_threads
                    zi = ((bz * gdy + by) * gdx + bx) * epi_threads + tidx
                    while zi < _HZB // 16:
                        h_st_zero16(z_base + Int64(zi) * Int64(16))
                        zi += nthr
                while work_tile.is_valid_tile:
                    cur = work_tile.tile_idx
                    m_t = cur[0]
                    n_t = cur[1]
                    l_t = cur[2]
                    cnt = tile_sched_params.masked_m[l_t]
                    n_valid = cnt - n_t * tile_n
                    if n_valid > tile_n:
                        n_valid = Int32(tile_n)
                    n_chunks = (n_valid + (_HEPI_COLS - 1)) // _HEPI_COLS
                    if cutlass.const_expr(_HNOEPI == 1):
                        n_chunks = Int32(0)
                    alpha_v = alpha[l_t]
                    p_ph = Int32(0)
                    n_chunks2 = Int32(0)
                    pre_row = Int32(0)
                    pre_w = cutlass.Float32(0.0)
                    if cutlass.const_expr(_PF):
                        p_ph = cur[3] & 1
                        if p_ph == 1:
                            alpha_v = alpha2[l_t]
                            n_chunks2 = n_chunks
                            n_chunks = Int32(0)
                            if tidx < tile_n:
                                pre_row = p_perm[(l_t, n_t * tile_n + tidx)]
                                pre_w = p_rw[(l_t, n_t * tile_n + tidx)] * alpha_v
                    if cutlass.const_expr(_PNOEPI):
                        n_chunks = Int32(0)
                        n_chunks2 = Int32(0)
                    j_role = Int32(0)
                    j_s = Int32(0)
                    j_p = Int32(0)
                    j_P1 = Int32(0)
                    if cutlass.const_expr(_JS):
                        j_w1 = cur[3]
                        j_s = j_w1 >> 16
                        j_p = (j_w1 >> 13) & 7
                        j_P1 = (j_w1 >> 10) & 7
                        if j_s > 0:
                            j_role = Int32(1)
                            if j_p == j_P1:
                                j_role = Int32(2)
                        if j_role == 2:
                            if tidx == 0:
                                j_spin = j_ld_acquire(j_ctr.iterator.toint() + 4 * (64 + j_s))
                                while j_spin < j_P1:
                                    j_spin = j_ld_acquire(j_ctr.iterator.toint() + 4 * (64 + j_s))
                            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epi_threads)
                            j_fence_gpu()
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    if warp_idx == 0:
                        if j_t < 16:
                            self.j_trace(j_order, bidz, 20 + j_t)
                    for ci in cutlass.range(n_chunks, unroll=1):
                        cute.copy(
                            tiled_t2r,
                            tTR_tAcc_h[(None, None, None, 0, ci, acc_consumer_state.index)],
                            tTR_r,
                        )
                        if cutlass.const_expr(_JS):
                            if j_role == 1:
                                for j in cutlass.range_constexpr(cute.size(tTR_r)):
                                    rr, cc = tTR_cId[j]
                                    j_ws[(((j_s * 8 + j_p) * 4 + ci) * 128 + rr) * 32 + cc] = tTR_r[j]
                            if j_role == 2:
                                for q in cutlass.range(j_P1, unroll=1):
                                    for j in cutlass.range_constexpr(cute.size(tTR_r)):
                                        rr, cc = tTR_cId[j]
                                        tTR_r[j] = tTR_r[j] + j_ws[(((j_s * 8 + q) * 4 + ci) * 128 + rr) * 32 + cc]
                        if j_role != 1:
                            row0 = tTR_cId[0][0]
                            if cutlass.const_expr(_HNOEPI == 2):
                                row0 = Int32(-1000)  # ablation: TMEM load only
                            if row0 >= 64:
                                for j in cutlass.range_constexpr(cute.size(tTR_r)):
                                    g = tTR_r[j] * alpha_v
                                    sg = g * cute.arch.rcp_approx(
                                        cutlass.Float32(1.0) + cute.math.exp2(-(g * LOG2E), fastmath=True)
                                    )
                                    rr, cc = tTR_cId[j]
                                    sEpi[(rr - 64, cc)] = sg
                            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epi_threads)
                            if row0 < 64:
                                for j in cutlass.range_constexpr(cute.size(tTR_r)):
                                    rr, cc = tTR_cId[j]
                                    sEpi[(rr, cc)] = tTR_r[j] * alpha_v * sEpi[(rr, cc)]
                            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epi_threads)
                            slot = n_t * tile_n + ci * _HEPI_COLS + q_c
                            if cutlass.const_expr(_HNOEPI == 2):
                                slot = Int32(1 << 20)
                            if slot < cnt and q_b < 4:
                                hv = [sEpi[(q_b * 16 + i, q_c)] for i in range(16)]
                                amax = cutlass.Float32(0.0)
                                for i in cutlass.range_constexpr(16):
                                    amax = cute.arch.fmax(amax, cute.arch.fmax(hv[i], -hv[i]))
                                code = h_e4m3_code(gs * (amax * rcp6))
                                sfd = h_e4m3_decode(code)
                                osc = cutlass.Float32(0.0)
                                if sfd != cutlass.Float32(0.0):
                                    osc = cute.arch.rcp_approx(sfd * inv_gs)
                                lo = h_cvt_e2m1x8(*[hv[i] * osc for i in range(8)])
                                hi = h_cvt_e2m1x8(*[hv[i] * osc for i in range(8, 16)])
                                h_out[(l_t, slot, m_t * 4 + q_b)] = (cutlass.Uint64(hi) << cutlass.Uint64(32)) | cutlass.Uint64(lo)
                                sf_off = (
                                    ((l_t * sf_rm + slot // 128) * sf_rk + m_t) * 32 + slot % 32
                                ) * 16 + ((slot % 128) // 32) * 4 + q_b
                                h_sf[sf_off] = code.to(cutlass.Uint8)
                            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epi_threads)
                    if cutlass.const_expr(_PF):
                        if p_ph == 1:
                            if cutlass.const_expr(_PZ):
                                if p_zready == 0:
                                    p_zslot = 2 * _PNG
                                    if cutlass.const_expr(_PZTL):
                                        p_zslot = 2 * _PNG + 7
                                        if p_zdone == 0:
                                            self.p_zero_share(p_out, p_gates, tidx)
                                            p_zdone = Int32(1)
                                    if tidx == 0:
                                        p_zv = j_ld_acquire(p_gaddr + Int64(4 * p_zslot))
                                        while p_zv < p_ztarget:
                                            p_zv = j_ld_acquire(p_gaddr + Int64(4 * p_zslot))
                                    cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epi_threads)
                                    p_zready = Int32(1)
                        for ci in cutlass.range(n_chunks2, unroll=1):
                            cute.copy(
                                tiled_t2r,
                                tTR_tAcc_h[(None, None, None, 0, ci, acc_consumer_state.index)],
                                tTR_r,
                            )
                            slot0 = n_t * tile_n + ci * _HEPI_COLS
                            rb = p_ndone % 2
                            if tidx // _HEPI_COLS == ci:
                                sW[(tidx % _HEPI_COLS, rb)] = pre_w
                                sR[(tidx % _HEPI_COLS, rb)] = pre_row
                            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epi_threads)
                            for j in cutlass.range_constexpr(cute.size(tTR_r)):
                                rr, cc = tTR_cId[j]
                                sEpi2[(cc, rr)] = (tTR_r[j] * sW[(cc, rb)]).to(cutlass.BFloat16)
                            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epi_threads)
                            for it in cutlass.range_constexpr(_HEPI_COLS // 8):
                                tok = it * 8 + tidx // 16
                                pc = tidx % 16
                                slot = slot0 + tok
                                if slot < cnt:
                                    row = sR[(slot - slot0, rb)]
                                    dst = p_gbase + (Int64(row) * Int64(4096) + Int64(m_t * 128 + pc * 8)) * Int64(2)
                                    sp = cute.make_ptr(cutlass.Uint32, p_sbase + tok * 272 + pc * 16,
                                                       cute.AddressSpace.smem, assumed_align=16)
                                    v4 = cute.make_tensor(sp, cute.make_layout(4)).load()
                                    p_red_bf16x8(dst, v4[0], v4[1], v4[2], v4[3])
                            p_ndone += 1
                        if p_ph == 0:
                            p_fence_proxy_async_global()
                            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epi_threads)
                            if tidx == 0:
                                p_red_release(p_gaddr + Int64(4) * Int64(p_bank * _PNG + l_t * 64 + n_t), Int32(1))
                    if cutlass.const_expr(_JS):
                        if j_role == 1:
                            j_fence_gpu()
                            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=epi_threads)
                            if tidx == 0:
                                atomic_add_release_global(j_ctr.iterator.toint() + 4 * (64 + j_s), Int32(1))
                        if j_role == 2:
                            if tidx == 0:
                                j_ctr[64 + j_s] = Int32(0)
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_consumer_state)
                    if warp_idx == 0:
                        if j_t < 16:
                            self.j_trace(j_order, bidz, 40 + j_t)
                    j_t += 1
                    acc_consumer_state.advance()
                    tile_sched.advance_to_next_work()
                    work_tile, _ = tile_sched.get_current_work()
                if cutlass.const_expr(_HZB > 0 and _HZPOS == 1):
                    # Lane I: zero the FC2 finalize output, which the caller placed right after the SF buffer
                    # (same allocation), after this CTA's last tile (tail).
                    z_base = Int64(h_sf.iterator.toint()) + Int64(cute.cosize(h_sf.layout))
                    gdx, gdy, gdz = cute.arch.grid_dim()
                    bx, by, bz = cute.arch.block_idx()
                    nthr = gdx * gdy * gdz * epi_threads
                    zi = ((bz * gdy + by) * gdx + bx) * epi_threads + tidx
                    while zi < _HZB // 16:
                        h_st_zero16(z_base + Int64(zi) * Int64(16))
                        zi += nthr
            else:
                #
                # Partition for epilogue
                #
                epi_tidx = tidx
                tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                    self.epilog_tmem_copy_and_partition(
                        epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
                    )
                )

                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC
                )
                tma_atom_c, bSG_sC, bSG_gC_partitioned = (
                    self.epilog_gmem_copy_and_partition(
                        epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                    )
                )

                # S2R copy for topk_weights: SMEM → registers with T2R-compatible layout
                if cutlass.const_expr(self.is_combine_fusion):
                    topk_s2r_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), self.acc_dtype
                    )
                    topk_tiled_s2r = cute.make_tiled_copy_D(topk_s2r_atom, tiled_copy_t2r)
                    topk_thr_s2r = topk_tiled_s2r.get_slice(epi_tidx)
                    topk_rFrag = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)
                    topk_rRetiled = topk_tiled_s2r.retile(topk_rFrag)

                #
                # Persistent tile scheduling loop
                #
                tile_sched = (UnitScheduler if _HS else MaskedScheduler).create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()

                acc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage
                )

                # Threads/warps participating in tma store pipeline
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp * len(self.epilog_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=c_producer_group,
                )

                if cutlass.const_expr(tile_sched_params.dst_signals is not None):
                    assert self.num_c_stage < 256, "must be representable in 1 byte"
                    num_experts = tile_sched_params.masked_m.shape[0]
                    assert num_experts <= 8, "need to be packable into a u64"
                dsm_pending_packed = Uint64(0)
                dsm_pending_idx = Int32(0)
                dsm_counter = Uint8(0)
                c_dtype_bytes: cutlass.Constexpr = self.c_dtype.width // 8
                epilog_threads = self.threads_per_warp * len(self.epilog_warp_id)

                while work_tile.is_valid_tile:
                    # Get tile coord from tile scheduler
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_mnl = (
                        cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                        cur_tile_coord[1],
                        cur_tile_coord[2],
                    )

                    #
                    # Slice to per mma tile index
                    #
                    # ((ATOM_V, REST_V), EPI_M, EPI_N)
                    bSG_gC = bSG_gC_partitioned[
                        (
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]

                    #
                    # Prefetch out_ptrs and topk_weights for this tile into SMEM
                    # (overlaps GMEM loads with MMA pipeline stall)
                    #
                    if cutlass.const_expr(self.is_combine_fusion):
                        prefetch_m_base = cur_tile_coord[1] * self.mma_inst_shape_mnk[1]
                        prefetch_num_m: cutlass.Constexpr = self.mma_inst_shape_mnk[1]
                        prefetch_l_idx = cur_tile_coord[2]
                        if tidx < prefetch_num_m:
                            prefetch_m_idx = prefetch_m_base + tidx
                            smem_topk[tidx] = topk_weights[prefetch_l_idx, prefetch_m_idx]
                            if prefetch_m_idx < tile_sched_params.masked_m[prefetch_l_idx]:
                                src_rank = rank_src_info[prefetch_l_idx, prefetch_m_idx]
                                src_idx = idx_src_info[prefetch_l_idx, prefetch_m_idx]
                                col_size = mC_mnl.shape[0]
                                smem_out_ptrs[tidx] = (
                                    out_ptrs[src_rank]
                                    + (src_idx * col_size) * c_dtype_bytes
                                )
                            else:
                                smem_out_ptrs[tidx] = Int64(0)

                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epilog_threads,
                        )

                    # Set tensor memory buffer for current tile
                    # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_consumer_state.index)
                    ]

                    #
                    # Wait for accumulator buffer full
                    #
                    acc_pipeline.consumer_wait(acc_consumer_state)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                    cur_tile_dim0_offset = cur_tile_coord[0] * self.mma_inst_shape_mnk[0]
                    cur_tile_dim1_offset = cur_tile_coord[1] * self.mma_inst_shape_mnk[1]
                    #
                    # Store accumulator to global memory in subtiles
                    #
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                    for subtile_idx in cutlass.range(subtile_cnt):
                        #
                        # Load accumulator from tensor memory buffer to register
                        #
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        #
                        # Convert to C type
                        #
                        if cutlass.const_expr(self.is_combine_fusion):
                            # S2R copy: load topk from SMEM into register fragment
                            topk_smem_offset = (
                                subtile_idx
                                * epi_tile[1].shape
                                * (self.acc_dtype.width // 8)
                            )
                            topk_stride = (0, 1)
                            subtile_topk_ptr = cute.make_ptr(
                                self.acc_dtype,
                                smem_topk.iterator.toint() + topk_smem_offset,
                                cute.AddressSpace.smem,
                            )
                            smem_topk_2d = cute.make_tensor(
                                subtile_topk_ptr,
                                cute.make_layout(
                                    (epi_tile[0].shape, epi_tile[1].shape),
                                    stride=topk_stride,
                                ),
                            )
                            tSR_sTopk = topk_thr_s2r.partition_S(smem_topk_2d)
                            cute.copy(topk_tiled_s2r, tSR_sTopk, topk_rRetiled)
                            topk_regs_mn = topk_rFrag.load()
                            acc_vec = tTR_rAcc.load()
                        else:
                            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        if cutlass.const_expr(alpha is not None):
                            acc_vec = acc_vec * alpha[work_tile.tile_idx[2]]

                        if cutlass.const_expr(self.is_combine_fusion):
                            acc_vec = acc_vec * topk_regs_mn

                        acc_vec = acc_vec.to(self.c_dtype)
                        tRS_rC.store(acc_vec)

                        #
                        # Store C to shared memory
                        #
                        c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epilog_threads,
                        )

                        #
                        # TMA store C to global memory
                        #
                        if cutlass.const_expr(self.is_combine_fusion):
                            src_smem_tensor = bSG_sC[(None, c_buffer)]
                            n_idx = cur_tile_dim0_offset
                            l_idx = cur_tile_coord[2]
                            with cute.arch.elect_one():
                                local_m = subtile_idx * epi_tile[1].shape + warp_idx * 8
                                smem_out_ptrs_ptr = cute.make_ptr(
                                    Int64,
                                    smem_out_ptrs.iterator.toint() + local_m * 8,
                                    cute.AddressSpace.smem,
                                )
                                reg_out_ptrs = cute.make_tensor(
                                    smem_out_ptrs_ptr, (8,)
                                ).load()
                                # 8 elements per warp for a total of 32 elements per subtile
                                for idx_in_warp in cutlass.range_constexpr(8):
                                    m_row = warp_idx * 8 + idx_in_warp
                                    m_idx = (
                                        cur_tile_dim1_offset
                                        + subtile_idx * epi_tile[1].shape
                                        + m_row
                                    )
                                    out_ptr = reg_out_ptrs[idx_in_warp]
                                    if m_idx < tile_sched_params.masked_m[l_idx]:
                                        src_ptr = cute.make_ptr(
                                            self.c_dtype,
                                            out_ptr + n_idx * c_dtype_bytes,
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        )
                                        if cutlass.const_expr(self.c_dtype.width == 16):
                                            cp_reduce_bf16_add(
                                                src_ptr,
                                                src_smem_tensor[((None, m_row),)].iterator,
                                                epi_tile[0].shape * c_dtype_bytes,
                                            )
                                        else:
                                            cp_reduce_f32_add(
                                                src_ptr,
                                                src_smem_tensor[((None, m_row),)].iterator,
                                                epi_tile[0].shape * c_dtype_bytes,
                                            )
                                # Fence and barrier to make sure shared memory store is visible to TMA store
                                c_pipeline.producer_commit()
                                c_pipeline.producer_acquire()
                        else:
                            if warp_idx == self.epilog_warp_id[0]:
                                if cutlass.const_expr(_GNOC == 0):
                                    cute.copy(
                                        tma_atom_c,
                                        bSG_sC[(None, c_buffer)],
                                        bSG_gC[(None, subtile_idx)],
                                    )
                                elif cutlass.const_expr(_GNOC == 2):
                                    if (
                                        cur_tile_dim1_offset + subtile_idx * epi_tile[1].shape
                                        < tile_sched_params.masked_m[cur_tile_coord[2]]
                                    ):
                                        cute.copy(
                                            tma_atom_c,
                                            bSG_sC[(None, c_buffer)],
                                            bSG_gC[(None, subtile_idx)],
                                        )
                                # Fence and barrier to make sure shared memory store is visible to TMA store
                                c_pipeline.producer_commit()

                                if cutlass.const_expr(
                                    tile_sched_params.dst_signals is not None
                                ):
                                    dsm_counter = (dsm_counter + 1).to(Uint8)
                                    will_write_signals = (
                                        read_byte(dsm_pending_packed, dsm_pending_idx)
                                        == dsm_counter
                                    )

                                    if will_write_signals:
                                        # The original c_pipeline.producer_acquire()
                                        #   := PipelineTmaStore.producer_acquire()
                                        #   := TmaStoreFence.wait()
                                        #   := cute.arch.cp_async_bulk_wait_group(self.num_stages - 1, read=True)
                                        cute.arch.cp_async_bulk_wait_group(
                                            self.num_c_stage - 1,
                                            # Change `read` from True to False to also wait writes
                                            read=False,
                                        )
                                    else:
                                        c_pipeline.producer_acquire()

                                else:
                                    c_pipeline.producer_acquire()

                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epilog_threads,
                        )

                        if cutlass.const_expr(tile_sched_params.dst_signals is not None):
                            lane_id = tidx % 32
                            if warp_idx == self.epilog_warp_id[0] and lane_id == 0:
                                while (dsm_pending_idx < num_experts) and (
                                    read_byte(dsm_pending_packed, dsm_pending_idx)
                                    == dsm_counter
                                ):
                                    atomic_add_release_global(
                                        tile_sched_params.dst_signals.toint()
                                        + sizeof_i32 * dsm_pending_idx,
                                        value=1,
                                    )
                                    dsm_pending_idx += 1

                    #
                    # Async arrive accumulator buffer empty
                    #
                    acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    #
                    # Advance to next tile
                    #
                    tile_sched.advance_to_next_work()
                    work_tile, dsm_pending_packed = tile_sched.get_current_work(
                        dsm_pending_packed=dsm_pending_packed,
                        dsm_counter=dsm_counter,
                        num_c_stage=self.num_c_stage,
                    )

            if cutlass.const_expr(_PF and _PZ and _PZTL):
                if warp_idx < self.mma_warp_id:
                    if p_zdone == 0:
                        self.p_zero_share(p_out, p_gates, tidx)
            if cutlass.const_expr(_PF and _PZ and _PZT):
                # tail: finished CTAs claim 4-token chunks and zero the rows with no local expert (DRAM idle here)
                zt_gb = Int64(p_gates.iterator.toint())
                zt_T = p_gates[2 * _PNG + 1] // 512
                zt_nch = (zt_T + 3) // 4
                zt_ob = Int64(p_out.iterator.toint())
                zt_go = Int32(1)
                while zt_go == 1:
                    if tidx == 0:
                        p_st_vol_shared(p_flag_addr + 12, p_atom_add(zt_gb + Int64(4 * (2 * _PNG + 6)) + Int64(4) * Int64(p_bank), Int32(1)))
                    cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=128)
                    zt_c = p_ld_vol_shared(p_flag_addr + 12)
                    cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=128)
                    if zt_c >= zt_nch:
                        zt_go = Int32(0)
                    else:
                        for zr in cutlass.range_constexpr(4):
                            zt_r = zt_c * 4 + zr
                            if zt_r < zt_T:
                                if p_gates[_PNL0 + zt_r] == 0:
                                    for zq in cutlass.range_constexpr(4):
                                        h_st_zero16(zt_ob + Int64(zt_r) * Int64(8192) + Int64((zq * 128 + tidx) * 16))
            if cutlass.const_expr(_PF and _PTR):
                if tidx == 0:
                    pgx3, pgy3, pgz3 = cute.arch.grid_dim()
                    pbx3, pby3, pbz3 = cute.arch.block_idx()
                    p_gates[2 * _PNG + 8 + 8 * ((pbz3 * pgy3 + pby3) * pgx3 + pbx3) + 2] = j_gtimer()
            #
            # Dealloc the tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )
            #
            # Wait for C store complete
            #
            if cutlass.const_expr(tile_sched_params.dst_signals is not None):
                # The original c_pipeline.producer_tail()
                #   := PipelineTmaStore.producer_tail()
                #   := TmaStoreFence.tail()
                #   := cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.cp_async_bulk_wait_group(
                    0,
                    # Change `read` from True to False to also wait writes
                    read=False,
                )

                lane_id = tidx % 32
                if warp_idx == self.epilog_warp_id[0] and lane_id == 0:
                    while dsm_pending_idx < num_experts:
                        atomic_add_release_global(
                            tile_sched_params.dst_signals.toint()
                            + sizeof_i32 * dsm_pending_idx,
                            value=1,
                        )
                        dsm_pending_idx += 1

            elif cutlass.const_expr(self.is_combine_fusion):
                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_wait_group(
                        0,
                        # Change `read` from True to False to also wait writes
                        read=False,
                    )
                cute.arch.barrier(
                    barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
                )
            elif cutlass.const_expr(not _HF):
                c_pipeline.producer_tail()

            if cutlass.const_expr(self.is_combine_fusion):
                if warp_idx == self.epilog_warp_id[0]:
                    with cute.arch.elect_one():
                        # Linear id of current SM.
                        sm_id_linear = (
                            cute.arch.block_idx()[0]
                            + cute.arch.block_idx()[1] * cute.arch.grid_dim()[0]
                            + cute.arch.block_idx()[2]
                            * cute.arch.grid_dim()[0]
                            * cute.arch.grid_dim()[1]
                        )
                        # Release flag with sys scope
                        utils.distributed.multimem_red_add1(
                            lock_ptr=barrier_flag_multicast.iterator + sm_id_linear,
                            scope="sys",
                            order="release",
                        )
                        # Relaxed spin-lock wait flag with sys scope
                        utils.distributed.spin_lock_atom_cas_relaxed_wait(
                            lock_ptr=barrier_flag_local.iterator + sm_id_linear,
                            expected_val=self.num_ranks,
                            reset_val=0,
                            scope="sys",
                        )

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tma_atom_c, bSG_sC, bSG_gC) where:
            - tma_atom_c: The TMA copy atom
            - bSG_sC: The partitioned shared memory tensor C
            - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        if cutlass.const_expr(self.is_combine_fusion):
            return tma_atom_c, sC_for_tma_partition, gC_for_tma_partition
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        a_major_mode: cute.nvgpu.OperandMajorMode,
        b_dtype: Type[cutlass.Numeric],
        b_major_mode: cute.nvgpu.OperandMajorMode,
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
        is_combine_fusion: bool = False,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param a_major_mode: Major mode of operand A.
        :type a_major_mode: cute.nvgpu.OperandMajorMode
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param b_major_mode: Major mode of operand B.
        :type b_major_mode: cute.nvgpu.OperandMajorMode
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of Scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor vector size.
        :type sf_vec_size: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_c_stage = _GC

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        if is_combine_fusion:
            c_smem_layout_staged_one = make_fused_smem_layout_epi(
                c_dtype,
                c_layout,
                epi_tile,
                1,
            )
        else:
            c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
                c_dtype,
                c_layout,
                epi_tile,
                1,
            )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        # Account for out_ptrs and topk_weights prefetch buffers
        if is_combine_fusion:
            num_m_rows = mma_tiler_mnk[1]
            combine_prefetch_bytes = (
                num_m_rows * 8  # sizeof(Int64) for out_ptrs
                + num_m_rows * 4  # sizeof(Float32) for topk_weights
            )
            mbar_helpers_bytes += combine_prefetch_bytes
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage
        if _HF and _HACC:
            num_acc_stage = _HACC
        if _HF:
            # fused SwiGLU+NVFP4 epilogue: no C staging; one fp32 (64 x 32, row stride 33) transpose buffer
            num_c_stage = 1
            c_bytes = max(64 * (_HEPI_COLS + 1) * 4, _HEPI_COLS * 272) + 16 * _HEPI_COLS + 128
            num_ab_stage = (smem_capacity - (mbar_helpers_bytes + c_bytes)) // ab_bytes_per_stage
            if _GAB:
                num_ab_stage = _GAB
            print('[H] stages', dict(mma_tiler=tuple(mma_tiler_mnk), ab_stage_bytes=ab_bytes_per_stage, num_ab_stage=num_ab_stage,
                  num_acc_stage=num_acc_stage, smem_used=num_ab_stage * ab_bytes_per_stage + c_bytes + mbar_helpers_bytes,
                  cap=smem_capacity, sched=_HS), flush=True)
            return num_acc_stage, num_ab_stage, num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        if _GAB:
            num_ab_stage = _GAB
        num_c_stage += 0 if not _GCR else (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        print('[G] stages', dict(mma_tiler=tuple(mma_tiler_mnk), ab_stage_bytes=ab_bytes_per_stage,
              a=cute.size_in_bytes(a_dtype, a_smem_layout_stage_one), b=cute.size_in_bytes(b_dtype, b_smem_layout_staged_one),
              sfa=cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one), sfb=cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one),
              c_stage_bytes=c_bytes_per_stage, num_ab_stage=num_ab_stage, num_c_stage=num_c_stage, num_acc_stage=num_acc_stage,
              smem_used=num_ab_stage*ab_bytes_per_stage+num_c_stage*c_bytes_per_stage+mbar_helpers_bytes, cap=smem_capacity), flush=True)
        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        masked_m_tensor: cute.Tensor,
        dst_signals: Optional[cute.Pointer],
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        is_swap_ab: cutlass.Constexpr,
    ) -> Tuple[MaskedSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[MaskedSchedulerParams, tuple[int, int, int]]
        """
        c_tiler = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = MaskedSchedulerParams(
            masked_m_tensor, dst_signals, c, c_tiler, cluster_shape_mnl, is_swap_ab
        )
        grid = MaskedScheduler.get_grid_shape(tile_sched_params, max_active_clusters)
        print('[P] grid', grid, 'cluster', cluster_shape_mn, 'max_active_clusters', max_active_clusters, flush=True)

        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [32, 64, 128, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor tensor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # Skip unsupported layouts
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L_mma_spec(
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)


# Create scale factor tensor SFA/SFB
def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype, device):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(k, sf_vec_size)
    ref_shape = (l, mn, sf_k)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    ref_permute_order = (1, 2, 0)
    mma_permute_order = (3, 4, 1, 5, 2, 0)

    # Create f32 ref torch tensor
    ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        ref_shape,
        torch.float32,
        permute_order=ref_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=1,
            max_val=3,
        ),
    )

    # Create f32 cute torch tensor
    cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=0,
            max_val=1,
        ),
    )

    # convert ref f32 tensor to cute f32 tensor
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )
    cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.to(device, non_blocking=True)

    # reshape makes memory contiguous
    ref_f32_torch_tensor_cpu = (
        ref_f32_torch_tensor_cpu.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, mn, sf_k, sf_vec_size)
        .reshape(l, mn, sf_k * sf_vec_size)
        .permute(*ref_permute_order)
    )
    # prune to mkl for reference check.
    ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]
    ref_f32_torch_tensor = ref_f32_torch_tensor_cpu.to(device, non_blocking=True)

    # Create dtype cute torch tensor (cpu)
    cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
        cute_f32_torch_tensor_cpu,
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )

    # Convert f32 cute tensor to dtype cute tensor
    cute_tensor = cutlass_torch.convert_cute_tensor(
        cute_f32_torch_tensor,
        cute_tensor,
        dtype,
        is_dynamic_layout=True,
    )
    return ref_f32_torch_tensor, cute_tensor, cute_torch_tensor


class MaskedBatchedMatmulCuteDSL:
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
        ab_dtype: torch.dtype,
        sf_dtype: torch.dtype,
        c_dtype: torch.dtype,
        alpha_dtype: torch.dtype,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        sm_count: int,
        sm_version: str,
        num_ranks: int = 0,
        is_combine_fusion: bool = False,
        is_swap_ab: bool = False,
    ):
        self._m = m
        self._n = n
        self._k = k
        self._l = l
        self._a_major = a_major
        self._b_major = b_major
        self._c_major = c_major
        self._ab_dtype = ab_dtype
        self._sf_dtype = sf_dtype
        self._c_dtype = c_dtype
        self._alpha_dtype = alpha_dtype
        self._sf_vec_size = sf_vec_size
        self._mma_tiler_mn = mma_tiler_mn
        self._cluster_shape_mn = cluster_shape_mn
        self._num_ranks = num_ranks
        self._num_sms = sm_count
        self._is_combine_fusion = is_combine_fusion
        self._is_swap_ab = is_swap_ab
        if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            l,
            a_major,
            b_major,
            c_major,
        ):
            raise TypeError(
                f"MaskedBatchedMatmulCuteDSL: Unsupported with {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype},  {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
            )

        # Compute max active clusters on current device (cached to avoid expensive queries)
        self._max_active_clusters = min(
            get_max_active_clusters(
                self._cluster_shape_mn[0] * self._cluster_shape_mn[1]
            ),
            sm_count,
        )
        self._sm_version = sm_version

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        masked_m_ptr: cute.Pointer,
        dst_signals_ptr: Optional[cute.Pointer],
        alpha_ptr: cute.Pointer,
        topk_weights_ptr: Optional[cute.Pointer],
        idx_src_info_ptr: Optional[cute.Pointer],
        rank_src_info_ptr: Optional[cute.Pointer],
        out_ptrs_ptr: Optional[cute.Pointer],
        barrier_flag_local_ptr: Optional[cute.Pointer],
        barrier_flag_multicast_ptr: Optional[cute.Pointer],
        h_out_ptr: Optional[cute.Pointer],
        h_sf_ptr: Optional[cute.Pointer],
        h_gs_ptr: Optional[cute.Pointer],
        j_order_ptr: Optional[cute.Pointer],
        j_ctr_ptr: Optional[cute.Pointer],
        j_ws_ptr: Optional[cute.Pointer],
        p_a2_ptr: Optional[cute.Pointer],
        p_sfa2_ptr: Optional[cute.Pointer],
        p_b2_ptr: Optional[cute.Pointer],
        p_sfb2_ptr: Optional[cute.Pointer],
        p_alpha2_ptr: Optional[cute.Pointer],
        p_out_ptr: Optional[cute.Pointer],
        p_perm_ptr: Optional[cute.Pointer],
        p_rw_ptr: Optional[cute.Pointer],
        p_gates_ptr: Optional[cute.Pointer],
        current_stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout(
                (self._m, self._k, self._l),
                order=(0, 1, 2) if self._a_major == "m" else (1, 0, 2),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (self._n, self._k, self._l),
                order=(0, 1, 2) if self._b_major == "n" else (1, 0, 2),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout(
                (self._m, self._n, self._l),
                order=(0, 1, 2) if self._c_major == "m" else (1, 0, 2),
            ),
        )
        row_size = self._n if self._c_major == "m" else self._m
        if cutlass.const_expr(self._is_combine_fusion):
            topk_weights_tensor = cute.make_tensor(
                topk_weights_ptr,
                layout=cute.make_ordered_layout((self._l, row_size), order=(1, 0)),
            )
            idx_src_info_tensor = (
                cute.make_tensor(
                    idx_src_info_ptr,
                    layout=cute.make_ordered_layout((self._l, row_size), order=(1, 0)),
                )
                if cutlass.const_expr(idx_src_info_ptr is not None)
                else None
            )
            rank_src_info_tensor = (
                cute.make_tensor(
                    rank_src_info_ptr,
                    layout=cute.make_ordered_layout((self._l, row_size), order=(1, 0)),
                )
                if cutlass.const_expr(rank_src_info_ptr is not None)
                else None
            )
            if cutlass.const_expr(rank_src_info_ptr is not None):
                out_ptrs_tensor = cute.make_tensor(
                    out_ptrs_ptr,
                    layout=cute.make_ordered_layout((self._num_ranks,), order=(0,)),
                )
            else:
                out_ptrs_tensor = cute.make_tensor(
                    out_ptrs_ptr,
                    layout=cute.make_ordered_layout((self._l, row_size), order=(1, 0)),
                )
            barrier_flag_local_tensor = cute.make_tensor(
                barrier_flag_local_ptr,
                layout=cute.make_ordered_layout((self._num_sms,), order=(0,)),
            )
            barrier_flag_multicast_tensor = cute.make_tensor(
                barrier_flag_multicast_ptr,
                layout=cute.make_ordered_layout((self._num_sms,), order=(0,)),
            )
        else:
            topk_weights_tensor = None
            idx_src_info_tensor = None
            rank_src_info_tensor = None
            out_ptrs_tensor = None
            barrier_flag_local_tensor = None
            barrier_flag_multicast_tensor = None

        # calculate sf_tensor shape and order
        def ceil_div(a, b):
            return (a + b - 1) // b

        sf_k = ceil_div(self._k, self._sf_vec_size)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape_a = (
            self._l,
            ceil_div(self._m, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )
        mma_shape_b = (
            self._l,
            ceil_div(self._n, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        sfa_tensor = cute.make_tensor(
            sfa_ptr,
            layout=cute.make_ordered_layout(
                mma_shape_a,
                order=mma_permute_order,
            ),
        )
        sfb_tensor = cute.make_tensor(
            sfb_ptr,
            layout=cute.make_ordered_layout(
                mma_shape_b,
                order=mma_permute_order,
            ),
        )
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L_mma_spec(sfa_tensor)
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L_mma_spec(sfb_tensor)

        masked_m_tensor = cute.make_tensor(
            masked_m_ptr,
            layout=cute.make_ordered_layout((self._l,), order=(0,)),
        )

        # Use const_expr for compile-time conditional
        alpha_tensor = (
            cute.make_tensor(
                alpha_ptr,
                layout=cute.make_ordered_layout((self._l,), order=(0,)),
            )
            if cutlass.const_expr(alpha_ptr is not None)
            else None
        )

        h_out_t = None
        h_sf_t = None
        h_gs_t = None
        if cutlass.const_expr(h_out_ptr is not None):
            h_out_t = cute.make_tensor(
                h_out_ptr,
                layout=cute.make_ordered_layout((self._l, self._n, self._m // 32), order=(2, 1, 0)),
            )
            h_sf_t = cute.make_tensor(
                h_sf_ptr,
                layout=cute.make_layout(self._l * ceil_div(self._n, 128) * (self._m // 32 // 4) * 512),
            )
            h_gs_t = cute.make_tensor(h_gs_ptr, layout=cute.make_layout(1))
        j_order_t = None
        j_ctr_t = None
        if cutlass.const_expr(j_order_ptr is not None):
            j_order_t = cute.make_tensor(j_order_ptr, layout=cute.make_layout(_JMAX))
            j_ctr_t = cute.make_tensor(j_ctr_ptr, layout=cute.make_layout(1024))
        j_ws_t = None
        if cutlass.const_expr(j_ws_ptr is not None):
            j_ws_t = cute.make_tensor(j_ws_ptr, layout=cute.make_layout(_JSMAX * 8 * 4 * 128 * 32 if _JS else 32))

        p_args = None
        if cutlass.const_expr(_PF):
            m2, k2 = self._k, self._m // 2  # FC2: M = hidden (= FC1 K), K = intermediate (= FC1 M / 2)
            a2_t = cute.make_tensor(p_a2_ptr, layout=cute.make_ordered_layout((m2, k2, self._l), order=(1, 0, 2)))
            b2_t = cute.make_tensor(p_b2_ptr, layout=cute.make_ordered_layout((self._n, k2, self._l), order=(1, 0, 2)))
            sf_k2 = ceil_div(k2, self._sf_vec_size)
            sfa2_t = cute.make_tensor(p_sfa2_ptr, layout=cute.make_ordered_layout(
                (self._l, ceil_div(m2, 128), ceil_div(sf_k2, 4), 32, 4, 4), order=mma_permute_order))
            sfb2_t = cute.make_tensor(p_sfb2_ptr, layout=cute.make_ordered_layout(
                (self._l, ceil_div(self._n, 128), ceil_div(sf_k2, 4), 32, 4, 4), order=mma_permute_order))
            cvt_sf_MKL_to_M32x4xrm_K4xrk_L_mma_spec(sfa2_t)
            cvt_sf_MKL_to_M32x4xrm_K4xrk_L_mma_spec(sfb2_t)
            p_args = (
                a2_t, b2_t, sfa2_t, sfb2_t,
                cute.make_tensor(p_alpha2_ptr, layout=cute.make_layout(self._l)),
                cute.make_tensor(p_out_ptr, layout=cute.make_ordered_layout((self._l * self._n, m2), order=(1, 0))),
                cute.make_tensor(p_perm_ptr, layout=cute.make_ordered_layout((self._l, self._n), order=(1, 0))),
                cute.make_tensor(p_rw_ptr, layout=cute.make_ordered_layout((self._l, self._n), order=(1, 0))),
                cute.make_tensor(p_gates_ptr, layout=cute.make_layout(2 * _PNG + 8)),
            )
        Sm100BlockScaledPersistentDenseGemmKernel(
            sf_vec_size=self._sf_vec_size,
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
            sm_version=self._sm_version,
            num_ranks=self._num_ranks,
            is_combine_fusion=self._is_combine_fusion,
            is_swap_ab=self._is_swap_ab,
        )(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            masked_m_tensor,
            dst_signals_ptr,
            alpha_tensor,
            topk_weights_tensor,
            idx_src_info_tensor,
            rank_src_info_tensor,
            out_ptrs_tensor,
            barrier_flag_local_tensor,
            barrier_flag_multicast_tensor,
            self._max_active_clusters,
            current_stream,
            h_out_t,
            h_sf_t,
            h_gs_t,
            j_order_t,
            j_ctr_t,
            j_ws_t,
            p_args,
        )


@functools.cache
def get_cute_dsl_compiled_masked_gemm_kernel(
    m: int,
    n: int,
    k: int,
    l: int,
    a_major: str,
    b_major: str,
    c_major: str,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    alpha_dtype: Optional[Type[cutlass.Numeric]],
    sf_vec_size: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    sm_count: int,
    sm_version: str,
    num_ranks: int,
    enable_dst_signals: bool,
    enable_barrier_flag: bool = False,
    is_combine_fusion: bool = False,
    is_swap_ab: bool = False,
) -> Callable:
    def get_cute_pointers(
        input_tensors: Optional[List[torch.tensor]],
    ) -> List[cute.Pointer]:
        if input_tensors is None:
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                masked_m_data_ptr,
                dst_signals_data_ptr,
                alpha_data_ptr,
                topk_weights_data_ptr,
                idx_src_info_data_ptr,
                rank_src_info_data_ptr,
                out_ptrs_data_ptr,
                barrier_flag_local_data_ptr,
                barrier_flag_multicast_data_ptr,
            ) = [16 for _ in range(14)]
            h_out_data_ptr, h_sf_data_ptr, h_gs_data_ptr = (16, 16, 16) if _HF else (None, None, None)
            j_order_data_ptr, j_ctr_data_ptr = (16, 16) if _JD else (None, None)
            j_ws_data_ptr = 16 if (_JS or _HPF) else None
            p_data = [16] * 9

            if not enable_dst_signals:
                dst_signals_data_ptr = None

            # When combine fusion is disabled, combine-related pointers should be None
            if not is_combine_fusion:
                topk_weights_data_ptr = None
                idx_src_info_data_ptr = None
                rank_src_info_data_ptr = None
                out_ptrs_data_ptr = None

            if not enable_barrier_flag:
                barrier_flag_local_data_ptr = None
                barrier_flag_multicast_data_ptr = None

        else:
            (
                a_tensor_gpu,
                b_tensor_gpu,
                sfa_tensor_gpu,
                sfb_tensor_gpu,
                c_tensor_gpu,
                masked_m_tensor_gpu,
                dst_signals_tensor_gpu,
                alpha_tensor_gpu,
                topk_weights_tensor_gpu,
                idx_src_info_tensor_gpu,
                rank_src_info_tensor_gpu,
                out_ptrs_tensor_gpu,
                barrier_flag_local_tensor_gpu,
                barrier_flag_multicast_tensor_gpu,
                h_out_gpu,
                h_sf_gpu,
                h_gs_gpu,
                *j_rest,
            ) = input_tensors
            j_order_data_ptr, j_ctr_data_ptr = (j_rest[0].data_ptr(), j_rest[1].data_ptr()) if _JD else (None, None)
            j_ws_data_ptr = j_rest[2].data_ptr() if (_JS or _HPF) else None
            p_data = [t.data_ptr() for t in j_rest[3:12]]
            h_out_data_ptr, h_sf_data_ptr, h_gs_data_ptr = (
                (h_out_gpu.data_ptr(), h_sf_gpu.data_ptr(), h_gs_gpu.data_ptr()) if _HF else (None, None, None)
            )

            assert enable_dst_signals == (dst_signals_tensor_gpu is not None)
            assert enable_barrier_flag == (barrier_flag_local_tensor_gpu is not None)

            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                masked_m_data_ptr,
                dst_signals_data_ptr,
                alpha_data_ptr,
            ) = (
                a_tensor_gpu.data_ptr(),
                b_tensor_gpu.data_ptr(),
                sfa_tensor_gpu.data_ptr(),
                sfb_tensor_gpu.data_ptr(),
                c_tensor_gpu.data_ptr(),
                masked_m_tensor_gpu.data_ptr(),
                dst_signals_tensor_gpu.data_ptr()
                if dst_signals_tensor_gpu is not None
                else None,
                alpha_tensor_gpu.data_ptr() if alpha_tensor_gpu is not None else None,
            )
            topk_weights_data_ptr = (
                topk_weights_tensor_gpu.data_ptr()
                if topk_weights_tensor_gpu is not None
                else None
            )
            idx_src_info_data_ptr = (
                idx_src_info_tensor_gpu.data_ptr()
                if idx_src_info_tensor_gpu is not None
                else None
            )
            rank_src_info_data_ptr = (
                rank_src_info_tensor_gpu.data_ptr()
                if rank_src_info_tensor_gpu is not None
                else None
            )
            out_ptrs_data_ptr = (
                out_ptrs_tensor_gpu.data_ptr()
                if out_ptrs_tensor_gpu is not None
                else None
            )
            barrier_flag_local_data_ptr = (
                barrier_flag_local_tensor_gpu.data_ptr()
                if barrier_flag_local_tensor_gpu is not None
                else None
            )
            barrier_flag_multicast_data_ptr = (
                barrier_flag_multicast_tensor_gpu.data_ptr()
                if barrier_flag_multicast_tensor_gpu is not None
                else None
            )

        a_ptr = make_ptr(
            ab_dtype,
            a_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        b_ptr = make_ptr(
            ab_dtype,
            b_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        sfa_ptr = make_ptr(
            sf_dtype,
            sfa_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        sfb_ptr = make_ptr(
            sf_dtype,
            sfb_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        c_ptr = make_ptr(
            c_dtype,
            c_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        masked_m_ptr = make_ptr(
            cutlass.Int32,
            masked_m_data_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        dst_signals_ptr = (
            make_ptr(
                cutlass.Uint32,
                dst_signals_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if dst_signals_data_ptr is not None
            else None
        )
        alpha_ptr = (
            make_ptr(
                alpha_dtype,
                alpha_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if alpha_data_ptr is not None and alpha_dtype is not None
            else None
        )
        topk_weights_ptr = (
            make_ptr(
                cutlass.Float32,
                topk_weights_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if topk_weights_data_ptr is not None
            else None
        )
        idx_src_info_ptr = (
            make_ptr(
                cutlass.Int32,
                idx_src_info_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if idx_src_info_data_ptr is not None
            else None
        )
        rank_src_info_ptr = (
            make_ptr(
                cutlass.Int32,
                rank_src_info_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if rank_src_info_data_ptr is not None
            else None
        )
        out_ptrs_ptr = (
            make_ptr(
                cutlass.Int64,
                out_ptrs_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if out_ptrs_data_ptr is not None
            else None
        )
        barrier_flag_local_ptr = (
            make_ptr(
                cutlass.Int32,
                barrier_flag_local_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if barrier_flag_local_data_ptr is not None
            else None
        )
        barrier_flag_multicast_ptr = (
            make_ptr(
                cutlass.Int32,
                barrier_flag_multicast_data_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if barrier_flag_multicast_data_ptr is not None
            else None
        )

        h_ptrs = [None, None, None, None, None, None]
        if h_out_data_ptr is not None:
            h_ptrs = [
                make_ptr(cutlass.Uint64, h_out_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Uint8, h_sf_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Float32, h_gs_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
                None, None, None,
            ]
        if j_order_data_ptr is not None:
            h_ptrs[3] = make_ptr(cutlass.Int32, j_order_data_ptr, cute.AddressSpace.gmem, assumed_align=16)
            h_ptrs[4] = make_ptr(cutlass.Int32, j_ctr_data_ptr, cute.AddressSpace.gmem, assumed_align=16)
        if j_ws_data_ptr is not None:
            h_ptrs[5] = make_ptr(cutlass.Float32 if _JS else cutlass.Int64, j_ws_data_ptr, cute.AddressSpace.gmem, assumed_align=16)
        return [
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            c_ptr,
            masked_m_ptr,
            dst_signals_ptr,
            alpha_ptr,
            topk_weights_ptr,
            idx_src_info_ptr,
            rank_src_info_ptr,
            out_ptrs_ptr,
            barrier_flag_local_ptr,
            barrier_flag_multicast_ptr,
        ] + h_ptrs + [
            make_ptr(ab_dtype, p_data[0], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, p_data[1], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(ab_dtype, p_data[2], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, p_data[3], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, p_data[4], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, p_data[5], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, p_data[6], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, p_data[7], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, p_data[8], cute.AddressSpace.gmem, assumed_align=16),
        ]

    kernel = cute.compile(
        MaskedBatchedMatmulCuteDSL(
            m=m,
            n=n,
            k=k,
            l=l,
            a_major=a_major,
            b_major=b_major,
            c_major=c_major,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            alpha_dtype=alpha_dtype,
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sm_count=sm_count,
            sm_version=sm_version,
            num_ranks=num_ranks,
            is_combine_fusion=is_combine_fusion,
            is_swap_ab=is_swap_ab,
        ),
        *get_cute_pointers(None),
        cutlass_torch.current_stream(),
    )

    def tensor_api(
        a_tensor_gpu: torch.Tensor,
        b_tensor_gpu: torch.Tensor,
        sfa_tensor_gpu: torch.Tensor,
        sfb_tensor_gpu: torch.Tensor,
        masked_m_tensor_gpu: torch.Tensor,
        dst_signals_tensor_gpu: torch.Tensor,
        c_tensor_gpu: Optional[torch.Tensor] = None,
        alpha_tensor_gpu: Optional[torch.Tensor] = None,
        topk_weights_tensor_gpu: Optional[torch.Tensor] = None,
        idx_src_info_tensor_gpu: Optional[torch.Tensor] = None,
        rank_src_info_tensor_gpu: Optional[torch.Tensor] = None,
        out_ptrs_tensor_gpu: Optional[torch.Tensor] = None,
        barrier_flag_local_tensor_gpu: Optional[torch.Tensor] = None,
        barrier_flag_multicast_tensor_gpu: Optional[torch.Tensor] = None,
        h_fused=(None, None, None),
    ):
        if c_tensor_gpu is None:
            # fp4 gemm output is not supported
            c_tensor_gpu = torch.empty(
                (l, m, n),
                dtype=cutlass_to_torch_dtype(c_dtype),
                device="cuda",
            )

        # fp4 or fp8 torch tensor to cute tensor
        current_stream = cutlass_torch.current_stream()

        nonlocal kernel
        kernel(
            *get_cute_pointers(
                [
                    a_tensor_gpu,
                    b_tensor_gpu,
                    sfa_tensor_gpu,
                    sfb_tensor_gpu,
                    c_tensor_gpu,
                    masked_m_tensor_gpu,
                    dst_signals_tensor_gpu,
                    alpha_tensor_gpu,
                    topk_weights_tensor_gpu,
                    idx_src_info_tensor_gpu,
                    rank_src_info_tensor_gpu,
                    out_ptrs_tensor_gpu,
                    barrier_flag_local_tensor_gpu,
                    barrier_flag_multicast_tensor_gpu,
                    *h_fused,
                ]
            ),
            current_stream,
        )

        return c_tensor_gpu

    return tensor_api


def _grouped_gemm_nt_masked_sm100(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    topk_weights: Optional[torch.Tensor] = None,
    idx_src_info: Optional[torch.Tensor] = None,
    rank_src_info: Optional[torch.Tensor] = None,
    out_ptrs: Optional[torch.Tensor] = None,
    num_ranks: int = 0,
    dst_signals: Optional[torch.Tensor] = None,
    sm_count: Optional[int] = None,
    barrier_flag_local: Optional[torch.Tensor] = None,
    barrier_flag_multicast: Optional[torch.Tensor] = None,
    is_combine_fusion: bool = False,
    is_swap_ab: bool = False,
    **kwargs,
):
    """
    SM100/SM103 (Blackwell) implementation of masked grouped GEMM.

    This is the arch-specific implementation; use ``grouped_gemm_nt_masked``
    from ``grouped_gemm_masked_wrapper`` as the public entry point.

    Executes a masked, batched matrix multiplication with scale factors and
    optional per-batch alpha scaling on the output.  ``alpha`` is currently
    applied internally by the kernel; see Notes for the canonical tensor
    layouts.

    Parameters
    ----------
    lhs : Tuple[torch.Tensor, torch.Tensor]
        ``(A, SFA)`` — left-hand-side input tensor and its scale-factor tensor.
        ``A`` has logical shape ``(m, k, l)`` (physically ``(l, m, k)``);
        for FP4 with 8-bit storage the physical shape is ``(m, k/2, l)``.
        ``SFA`` has logical shape ``(m32, m4, rm, k4, rk, l)``
        (physically ``(l, rm, rk, m32, m4, k4)``).
    rhs : Tuple[torch.Tensor, torch.Tensor]
        ``(B, SFB)`` — right-hand-side input tensor and its scale-factor
        tensor.  ``B`` has logical shape ``(n, k, l)``
        (physically ``(l, n, k)``; FP4 with 8-bit storage is
        ``(n, k/2, l)``).  ``SFB`` has logical shape
        ``(n32, n4, rn, k4, rk, l)``
        (physically ``(l, rn, rk, n32, n4, k4)``).
    out : torch.Tensor
        Output tensor of shape ``(l, m, n)``.  Mutated in place.
    masked_m : torch.Tensor
        1-D ``int32`` tensor of shape ``(l,)`` giving the valid row count of
        each batch.  Rows above ``masked_m[batch]`` are ignored.
    ab_dtype : str
        Data type for ``A`` and ``B``.  One of ``"float4_e2m1fn"``,
        ``"float8_e4m3fn"``, ``"float8_e5m2"``.
    sf_dtype : str
        Data type for the scale factors.  One of ``"float8_e8m0fnu"`` or
        ``"float8_e4m3fn"``.
    c_dtype : str
        Data type for output matrix ``C``.  One of ``"float16"``,
        ``"bfloat16"``, ``"float32"``, ``"float8_e4m3fn"``, ``"float8_e5m2"``.
    sf_vec_size : int
        Vector size for scale factors (typically 16 or 32).
    topk_weights : Optional[torch.Tensor]
        2-D ``float32`` tensor of shape ``(l, m)`` containing top-k routing
        weights.  Defaults to ``None``.
    idx_src_info : Optional[torch.Tensor]
        2-D ``int32`` tensor of shape ``(l, m)`` carrying source-index metadata
        for the combine fusion path.  Defaults to ``None``.
    rank_src_info : Optional[torch.Tensor]
        2-D ``int32`` tensor of shape ``(l, m)`` carrying rank-source metadata
        for the combine fusion path.  Defaults to ``None``.
    out_ptrs : Optional[torch.Tensor]
        1-D ``int64`` tensor of shape ``(num_ranks,)`` containing remote output
        pointers for multi-rank combine.  Defaults to ``None``.
    num_ranks : int
        Number of ranks participating in the combine path.  Defaults to ``0``.
    dst_signals : Optional[torch.Tensor]
        Optional 1-D signal tensor used by the combine-fusion path to notify
        consumers.  Defaults to ``None``.
    sm_count : Optional[int]
        Number of SMs to use.  If ``None``, the runtime picks the max available
        under the CTA configuration.
    barrier_flag_local : Optional[torch.Tensor]
        1-D ``int32`` tensor of shape ``(sm_count,)`` containing flags for
        local barrier synchronization (spin-lock wait in multi-rank ops).
        Defaults to ``None``.
    barrier_flag_multicast : Optional[torch.Tensor]
        1-D ``int32`` tensor of shape ``(sm_count,)`` containing flags for
        multicast barrier synchronization (release across ranks).  Defaults to
        ``None``.
    is_combine_fusion : bool
        If ``True``, enable the fused GEMM + combine operation mode.  Defaults
        to ``False``.
    is_swap_ab : bool
        If ``True``, swap the ``lhs``/``rhs`` input tensors.  Defaults to
        ``False``.
    **kwargs
        Additional keyword arguments.  Currently recognized:

        * ``mma_tiler_mn`` (``Tuple[int, int]``): shape of the MMA tiler
          ``(M, N)``.  Defaults to ``(128, 128)``.  ``mma_tiler_mn[0] == 256``
          enables the 2-CTA MMA path.  Must be ``(128, 128)`` when
          ``is_combine_fusion=True``.
        * ``cluster_shape_mn`` (``Tuple[int, int]``): shape of the CTA
          cluster ``(ClusterM, ClusterN)``.  Defaults to ``(1, 1)``.
        * ``alpha`` (``Optional[torch.Tensor]``): optional 1-D tensor of
          shape ``(l,)`` containing per-batch scaling factors.  When
          provided, each batch output is multiplied by its corresponding
          alpha value: ``out = alpha * (A @ B)``.
        * ``alpha_dtype`` (``str``): elemental dtype string for the
          ``alpha`` tensor (e.g. ``"float32"``).  Required when ``alpha``
          is provided.

        Other entries are reserved for forward-compatible kernel options.

    Notes
    -----
    Tensor-layout conventions:

    * ``l`` is the batch size, ``m``/``n`` are the row/column counts, and
      ``k`` is the contraction dimension.
    * ``m32`` and ``n32`` are the constants ``32``; ``m4`` and ``n4`` are the
      constants ``4``; and ``k4`` is the constant ``4``.
    * ``m32 * m4 * rm`` equals ``M``, where ``M`` is ``m`` padded up to the
      nearest multiple of 128.
    * ``n32 * n4 * rn`` equals ``N``, where ``N`` is ``n`` padded up to the
      nearest multiple of 128.
    * ``k4 * rk`` equals ``K``, where ``K`` is ``k / sf_vec_size`` padded up to
      the nearest multiple of 4.

    Masking is applied per batch via ``masked_m``.  When ``alpha`` is
    provided (see ``**kwargs``), each batch output is multiplied by its
    corresponding alpha value: ``out = alpha * (A @ B)``.
    """

    if is_swap_ab:
        a_torch, sfa_torch = rhs
        b_torch, sfb_torch = lhs
    else:
        a_torch, sfa_torch = lhs
        b_torch, sfb_torch = rhs

    c_torch = out

    # The (M, N)-tile TMA C store requires the output contiguous in the c_major
    # dim; an expert-innermost output (e.g. torch.empty(m, n, l)) silently
    # corrupts results across experts, so reject it up front (issue #3103).
    # The combine-fusion path uses a 2D scatter output (not the (M, N) TMA
    # store), so this layout contract does not apply there.
    if not is_combine_fusion:
        _c_major = "m" if is_swap_ab else "n"
        _c_perm = (2, 1, 0) if _c_major == "m" else (2, 0, 1)
        if out.dim() != 3 or not out.permute(*_c_perm).is_contiguous():
            raise ValueError(
                "grouped_gemm_nt_masked: `out` must be a 3D (m, n, l) tensor that is "
                f"contiguous in the layout implied by c_major='{_c_major}' (the "
                f"'{_c_major}' dimension contiguous and the batch/expert dim "
                "outermost). For c_major='n' allocate e.g. "
                "`torch.empty(l, m, n, ...).permute(1, 2, 0)`. Got out.shape="
                f"{tuple(out.shape)}, out.stride()={tuple(out.stride())}. A "
                "non-compliant layout (e.g. the expert dim innermost) causes silent "
                "cross-expert output corruption."
            )

    m, k, l = a_torch.shape
    n, _, _ = b_torch.shape

    if ab_dtype == "float4_e2m1fn":
        # todo(yingyi): update mnk based on a_major and b_major, and support more major.
        # Note: only support deepgemm-like shape for now
        k = k * 2

    mma_tiler_mn = kwargs.pop("mma_tiler_mn", (128, 128))
    cluster_shape_mn = kwargs.pop("cluster_shape_mn", (1, 1))
    if sm_count is None:
        sm_count = get_num_sm(a_torch.device)

    alpha = kwargs.pop("alpha", None)
    h_fused = kwargs.pop("h_fused", (None, None, None))
    alpha_dtype = kwargs.pop("alpha_dtype", None)

    assert len(kwargs) == 0, f"Unsupported kwargs: {kwargs}"

    major, minor = get_compute_capability(a_torch.device)
    if major == 11 and minor == 0:
        raise ValueError("SM110 is not supported for cute-dsl backend.")

    if is_combine_fusion:
        required = {
            "topk_weights": topk_weights,
            "idx_src_info": idx_src_info,
            "rank_src_info": rank_src_info,
            "out_ptrs": out_ptrs,
            "barrier_flag_local": barrier_flag_local,
            "barrier_flag_multicast": barrier_flag_multicast,
        }
        missing = [name for name, val in required.items() if val is None]
        if missing:
            raise ValueError(
                "is_combine_fusion=True requires non-None values for: "
                f"{', '.join(missing)}"
            )
        if num_ranks <= 0:
            raise ValueError("is_combine_fusion=True requires num_ranks > 0")
        # For combine fusion, we always swap the AB input tensors
        if not is_swap_ab:
            raise ValueError("is_combine_fusion=True requires is_swap_ab=True")
        if mma_tiler_mn != (128, 128):
            raise ValueError("is_combine_fusion=True requires mma_tiler_mn=(128, 128)")

    return get_cute_dsl_compiled_masked_gemm_kernel(
        m=m,
        n=n,
        k=k,
        l=l,
        a_major="k",
        b_major="k",
        c_major="m" if is_swap_ab else "n",
        ab_dtype=get_cutlass_dtype(ab_dtype),
        sf_dtype=get_cutlass_dtype(sf_dtype),
        c_dtype=get_cutlass_dtype(c_dtype),
        alpha_dtype=None if alpha is None else get_cutlass_dtype(alpha_dtype),
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sm_count=sm_count,
        sm_version=f"sm_{major}{minor}",
        num_ranks=num_ranks,
        enable_dst_signals=dst_signals is not None,
        enable_barrier_flag=barrier_flag_local is not None,
        is_combine_fusion=is_combine_fusion,
        is_swap_ab=is_swap_ab,
    )(
        a_tensor_gpu=a_torch,
        b_tensor_gpu=b_torch,
        sfa_tensor_gpu=sfa_torch,
        sfb_tensor_gpu=sfb_torch,
        c_tensor_gpu=c_torch,
        masked_m_tensor_gpu=masked_m,
        dst_signals_tensor_gpu=dst_signals,
        alpha_tensor_gpu=alpha,
        topk_weights_tensor_gpu=topk_weights,
        idx_src_info_tensor_gpu=idx_src_info,
        rank_src_info_tensor_gpu=rank_src_info,
        out_ptrs_tensor_gpu=out_ptrs,
        barrier_flag_local_tensor_gpu=barrier_flag_local,
        barrier_flag_multicast_tensor_gpu=barrier_flag_multicast,
        h_fused=h_fused,
    )


# Backwards compatibility: ``grouped_gemm_nt_masked`` used to live in this module
# and moved to ``grouped_gemm_masked_wrapper`` when SM107 support was added, since
# it now dispatches on compute capability rather than being Blackwell-specific.
# Re-exported here so the previous import path keeps working; prefer importing it
# from ``flashinfer.gemm``. The import is deferred to module scope only (the
# wrapper does not import this module at import time), so it introduces no cycle.
from flashinfer.gemm.kernels.grouped_gemm_masked_wrapper import (  # noqa: E402
    grouped_gemm_nt_masked as grouped_gemm_nt_masked,
)
