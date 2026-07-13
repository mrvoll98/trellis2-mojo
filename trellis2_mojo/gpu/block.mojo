# WP11 step 10: whole-block GPU residency for the DiT cross-block
# (dense ss_flow AND sparse slat share the exact same structure on flat
# feats [rows, C]). The running x stays device-resident through
#
#   ln+modulate -> self chain -> gate_add -> ln(affine) -> cross chain
#   -> add -> ln+modulate -> mlp chain -> gate_add
#
# with ONE x upload, ONE barrier and ONE readback per block instead of
# six transfers + CPU glue (~27 ms/block at ss geometry). All host
# uploads (x, glue consts, rope phases, cross kv) happen BEFORE any
# enqueue — mapping a host-written buffer commits pending queue work
# (gpu/linear.mojo law 2), so a mid-queue upload would silently
# serialize the queue. The per-block glue consts (shift/scale/gate pairs,
# the affine norm2 weights and the three out-biases folded into the adds)
# ride in ONE `bk` buffer indexed by an Int offset scalar — offset
# pointers are untested against the marshalling laws, offsets-as-scalars
# are free.
#
# Numerics: the layer norms mirror LayerNorm32's per-row formula (biased
# variance, 1/sqrt(var + eps)) with serial accumulation instead of the
# CPU's SIMD-tree -> tolerance parity (established); modulate/gate/bias
# adds are the exact same elementwise expressions as the CPU path. eps is
# baked comptime to the blocks' 1e-6 — the dispatch gate checks it.

from std.algorithm import parallelize
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceBuffer
from std.math import sqrt
from std.memory import memcpy

from trellis2_mojo.gpu.context import GpuContext
from trellis2_mojo.gpu.linear import GpuLinear, gpu_mlp_enqueue
from trellis2_mojo.gpu.attention import (
    GpuAttnChain,
    _attn_chain_enqueue,
    _cross_chain_enqueue,
    _cross_pack_kv,
    _upload_phases,
)
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32
comptime BM = 64
comptime LN_EPS = Float32(1e-6)  # the DiT block norms; dispatch gate checks


def ln_mod_rows(
    x: UnsafePointer[Scalar[F32], MutAnyOrigin],
    dst: UnsafePointer[Scalar[F32], MutAnyOrigin],
    consts: UnsafePointer[Scalar[F32], MutAnyOrigin],
    rows: Int, c: Int, off: Int, mode: Int,
):
    """Per-row layer norm (biased var, eps comptime 1e-6) fused with the
    adaLN epilogue. mode 1: modulate — consts[off:off+C] = shift,
    [off+C:off+2C] = scale, dst = ln(x)*(1+scale)+shift. mode 2: affine —
    consts pair = weight, bias, dst = ln(x)*w + b. One thread per row."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= rows:
        return
    var base = r * c
    var mean: Float32 = 0
    for i in range(c):
        mean += x[base + i]
    mean /= Float32(c)
    var variance: Float32 = 0
    for i in range(c):
        var dv = x[base + i] - mean
        variance += dv * dv
    variance /= Float32(c)
    var inv_std = 1.0 / sqrt(variance + LN_EPS)
    if mode == 1:
        for i in range(c):
            var nv = (x[base + i] - mean) * inv_std
            dst[base + i] = nv * (1.0 + consts[off + c + i]) + consts[off + i]
    else:
        for i in range(c):
            var nv = (x[base + i] - mean) * inv_std
            dst[base + i] = nv * consts[off + i] + consts[off + c + i]


def gate_add_bias_rows(
    x: UnsafePointer[Scalar[F32], MutAnyOrigin],
    h: UnsafePointer[Scalar[F32], MutAnyOrigin],
    consts: UnsafePointer[Scalar[F32], MutAnyOrigin],
    rows: Int, c: Int, off: Int, use_gate: Int,
):
    """Residual join with the chain's un-added out-bias folded in:
    x += (h + bias) * gate (use_gate 1) or x += h + bias (use_gate 0).
    consts[off:off+C] = bias, [off+C:off+2C] = gate. One thread per
    element; same elementwise expression as the CPU linear-bias + gate +
    residual sequence."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= rows * c:
        return
    var ci = i % c
    var hv = h[i] + consts[off + ci]
    if use_gate == 1:
        x[i] = x[i] + hv * consts[off + c + ci]
    else:
        x[i] = x[i] + hv


def gpu_block_phases(
    g: GpuContext, phases: Tensor[F32], rows: Int, d: Int
) raises -> DeviceBuffer[F32]:
    """Upload rope phases once per model forward (WP11 step 12 — every
    block in a forward shares them)."""
    return _upload_phases(g, phases, rows, d)


def gpu_block_state_upload(g: GpuContext, x: Tensor[F32], rows: Int, c: Int) raises:
    """Ensure the resident block state (xs/hs) and upload x into xs.
    Mapping the host-written buffer commits pending queue work — callers
    upload BEFORE enqueueing."""
    var mp = ((rows + BM - 1) // BM) * BM
    var s = g.scratch
    if s[].xs_cap < mp * c:
        s[].xs = g.ctx.enqueue_create_buffer[F32](mp * c)
        s[].xs_cap = mp * c
    if s[].hs_cap < mp * c:
        s[].hs = g.ctx.enqueue_create_buffer[F32](mp * c)
        s[].hs_cap = mp * c
    comptime NCHUNK = 16
    with s[].xs.value().map_to_host() as hm:
        var ap = hm.unsafe_ptr()
        var xp = x.data.unsafe_ptr()
        var total = rows * c
        var chunk = (total + NCHUNK - 1) // NCHUNK

        @parameter
        def copy_in(i: Int):
            var lo = i * chunk
            var n = min(chunk, total - lo)
            if n > 0:
                memcpy(dest=ap + lo, src=xp + lo, count=n)

        parallelize[copy_in](NCHUNK)


def gpu_block_state_readback(
    g: GpuContext, shape: List[Int], rows: Int, c: Int
) raises -> Tensor[F32]:
    """Read the resident state back to a host tensor (plain copy — the
    block path folds every bias in). Mapping xs (host-written at upload)
    commits + waits, so no explicit barrier is required first."""
    var out = Tensor[F32](shape)
    var s = g.scratch
    comptime NCHUNK = 16
    var op = out.data.unsafe_ptr()
    with s[].xs.value().map_to_host() as hm:
        var hp = hm.unsafe_ptr()
        var total = rows * c
        var chunk = (total + NCHUNK - 1) // NCHUNK

        @parameter
        def copy_out(i: Int):
            var lo = i * chunk
            var n = min(chunk, total - lo)
            if n > 0:
                memcpy(dest=op + lo, src=hp + lo, count=n)

        parallelize[copy_out](NCHUNK)
    return out^


def gpu_cross_block_forward(
    x: Tensor[F32],
    m: List[Tensor[F32]],
    k: Tensor[F32],
    v: Tensor[F32],
    use_rope: Bool,
    phases: Tensor[F32],
    norm2_w: Tensor[F32],
    norm2_b: Tensor[F32],
    self_chain: GpuAttnChain,
    self_qkv: GpuLinear,
    self_out: GpuLinear,
    cross_chain: GpuAttnChain,
    cross_q: GpuLinear,
    cross_out: GpuLinear,
    mlp0: GpuLinear,
    mlp2: GpuLinear,
) raises -> Tensor[F32]:
    """The whole ModulatedTransformerCrossBlock forward device-resident.
    x [.., C] flat rows; m = the six adaLN chunks [1, C]; k/v = the
    CPU-computed (and k-rms-normalized) cross kv [Lkv, H, D]. Callers
    gate on all three chains' wants + the norm layout first."""
    var c = self_chain.h * self_chain.d
    var ci = x.shape[len(x.shape) - 1]
    if ci != c:
        raise Error("gpu_cross_block_forward: channel mismatch")
    var rows = x.numel() // ci
    var g = self_chain.g.copy()
    gpu_block_state_upload(g, x, rows, c)
    var ph_buf = self_chain.consts
    if use_rope:
        ph_buf = _upload_phases(g, phases, rows, self_chain.d)
    gpu_cross_block_enqueue(
        rows, m, k, v, use_rope, ph_buf, norm2_w, norm2_b,
        self_chain, self_qkv, self_out, cross_chain, cross_q, cross_out,
        mlp0, mlp2,
    )
    g.barrier()
    return gpu_block_state_readback(g, x.shape, rows, c)


def gpu_cross_block_enqueue(
    rows: Int,
    m: List[Tensor[F32]],
    k: Tensor[F32],
    v: Tensor[F32],
    use_rope: Bool,
    ph_buf: DeviceBuffer[F32],
    norm2_w: Tensor[F32],
    norm2_b: Tensor[F32],
    self_chain: GpuAttnChain,
    self_qkv: GpuLinear,
    self_out: GpuLinear,
    cross_chain: GpuAttnChain,
    cross_q: GpuLinear,
    cross_out: GpuLinear,
    mlp0: GpuLinear,
    mlp2: GpuLinear,
) raises:
    """One block against the RESIDENT xs state (WP11 step 12): upload the
    per-block glue consts + host-pack the cross kv (these maps commit and
    wait out the previous block's kernels — the per-block sync in the
    model-resident loop), then enqueue the whole block. Callers ensure
    xs/hs via gpu_block_state_upload first."""
    var h = self_chain.h
    var d = self_chain.d
    var c = h * d
    if len(m) != 6:
        raise Error("gpu_cross_block_enqueue: expected 6 mod chunks")
    var mp = ((rows + BM - 1) // BM) * BM
    var lkv = k.shape[0]
    var lp = ((lkv + 63) // 64) * 64

    var g = self_chain.g.copy()
    var s = g.scratch
    if s[].bk_cap < 12 * c:
        s[].bk = g.ctx.enqueue_create_buffer[F32](12 * c)
        s[].bk_cap = 12 * c

    # ---- all host uploads BEFORE any enqueue (map commits the queue) ----
    # glue consts: 6 pairs of [C]+[C] at offsets 0, 2C, ... 10C:
    #   0: shift1, scale1        (norm1 modulate)
    #   1: self_out bias, gate1  (self residual join)
    #   2: norm2 weight, bias    (affine norm2)
    #   3: cross_out bias, 0     (cross residual join, ungated)
    #   4: shift3, scale3        (norm3 modulate)
    #   5: mlp2 bias, gate3      (mlp residual join)
    with s[].bk.value().map_to_host() as hb:
        var bp = hb.unsafe_ptr()
        memcpy(dest=bp, src=m[0].data.unsafe_ptr(), count=c)
        memcpy(dest=bp + c, src=m[1].data.unsafe_ptr(), count=c)
        if self_out.has_bias:
            memcpy(dest=bp + 2 * c, src=self_out.bias_host.unsafe_ptr(), count=c)
        else:
            for i in range(c):
                bp[2 * c + i] = 0
        memcpy(dest=bp + 3 * c, src=m[2].data.unsafe_ptr(), count=c)
        memcpy(dest=bp + 4 * c, src=norm2_w.data.unsafe_ptr(), count=c)
        memcpy(dest=bp + 5 * c, src=norm2_b.data.unsafe_ptr(), count=c)
        if cross_out.has_bias:
            memcpy(dest=bp + 6 * c, src=cross_out.bias_host.unsafe_ptr(), count=c)
        else:
            for i in range(c):
                bp[6 * c + i] = 0
        for i in range(c):
            bp[7 * c + i] = 0
        memcpy(dest=bp + 8 * c, src=m[3].data.unsafe_ptr(), count=c)
        memcpy(dest=bp + 9 * c, src=m[4].data.unsafe_ptr(), count=c)
        if mlp2.has_bias:
            memcpy(dest=bp + 10 * c, src=mlp2.bias_host.unsafe_ptr(), count=c)
        else:
            for i in range(c):
                bp[10 * c + i] = 0
        memcpy(dest=bp + 11 * c, src=m[5].data.unsafe_ptr(), count=c)

    _cross_pack_kv(g, k, v, h, d, lp)

    # ---- one in-order queue for the whole block ----
    var xs = s[].xs.value()
    var hs = s[].hs.value()
    var bk = s[].bk.value()
    # norm1 + modulate: xs -> hs
    g.ctx.enqueue_function[ln_mod_rows](
        xs.unsafe_ptr(), hs.unsafe_ptr(), bk.unsafe_ptr(),
        rows, c, 0, 1,
        grid_dim=((rows + 255) // 256,), block_dim=(256,),
    )
    # self-attention chain: hs -> hs
    _attn_chain_enqueue(
        self_chain, g, hs, rows, self_qkv, self_out, use_rope, ph_buf, hs
    )
    # xs += (hs + self_out_bias) * gate1
    g.ctx.enqueue_function[gate_add_bias_rows](
        xs.unsafe_ptr(), hs.unsafe_ptr(), bk.unsafe_ptr(),
        rows, c, 2 * c, 1,
        grid_dim=((rows * c + 255) // 256,), block_dim=(256,),
    )
    # affine norm2: xs -> hs
    g.ctx.enqueue_function[ln_mod_rows](
        xs.unsafe_ptr(), hs.unsafe_ptr(), bk.unsafe_ptr(),
        rows, c, 4 * c, 2,
        grid_dim=((rows + 255) // 256,), block_dim=(256,),
    )
    # cross-attention chain: hs -> hs (pre-packed ckt/cvh)
    _cross_chain_enqueue(
        cross_chain, g, hs, rows, lkv, cross_q, cross_out, hs
    )
    # xs += hs + cross_out_bias (ungated)
    g.ctx.enqueue_function[gate_add_bias_rows](
        xs.unsafe_ptr(), hs.unsafe_ptr(), bk.unsafe_ptr(),
        rows, c, 6 * c, 0,
        grid_dim=((rows * c + 255) // 256,), block_dim=(256,),
    )
    # norm3 + modulate: xs -> hs
    g.ctx.enqueue_function[ln_mod_rows](
        xs.unsafe_ptr(), hs.unsafe_ptr(), bk.unsafe_ptr(),
        rows, c, 8 * c, 1,
        grid_dim=((rows + 255) // 256,), block_dim=(256,),
    )
    # mlp chain: hs -> hs
    gpu_mlp_enqueue(g, hs, rows, mlp0, mlp2, hs)
    # xs += (hs + mlp2_bias) * gate3
    g.ctx.enqueue_function[gate_add_bias_rows](
        xs.unsafe_ptr(), hs.unsafe_ptr(), bk.unsafe_ptr(),
        rows, c, 10 * c, 1,
        grid_dim=((rows * c + 255) // 256,), block_dim=(256,),
    )
