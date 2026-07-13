# WP11: tiled f32 GEMM on the Metal GPU (threadgroup tiles + per-thread
# 4x4 register blocks) at the real DiT shapes, vs the CPU packed-GEMM
# linear (830 GF/s). Requires M%64 == N%64 == K%16 == 0 (the real wiring
# in trellis2_mojo/gpu/linear.mojo pads M and folds the bias into K).
# Not wired into pixi tasks.
#
# TIMING (fixed 2026-07-10): in 1.0.0b2-Metal, enqueue_function only
# stages work and ctx.synchronize() does NOT commit it — the only reliable
# commit+wait barrier is map_to_host of a host-written buffer. The
# original per-iteration enqueue+synchronize timing happened to measure a
# one-iteration-shifted window (each enqueue commits its predecessor), so
# the step-1 numbers were roughly right; this version times the honest
# window: enqueue -> flush-map wait. See gpu/linear.mojo header for the
# full set of empirical Metal laws.

from std.gpu import thread_idx, block_idx, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.time import perf_counter_ns

comptime BM = 64
comptime BN = 64
comptime BK = 16
comptime TM = 4
comptime TN = 4


def gemm_tiled(
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int, n: Int, kdim: Int,
):
    var As = stack_allocation[
        BM * BK, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * BN, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)  # 0..15 -> col group
    var ty = Int(thread_idx.y)  # 0..15 -> row group
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * BM + ty * TM
    var col0 = Int(block_idx.x) * BN + tx * TN
    var acc0 = SIMD[DType.float32, TN](0)
    var acc1 = SIMD[DType.float32, TN](0)
    var acc2 = SIMD[DType.float32, TN](0)
    var acc3 = SIMD[DType.float32, TN](0)
    var n_kb = kdim // BK
    for kb in range(n_kb):
        # cooperative loads: 256 threads x 4 contiguous elements each
        var ia = tid * 4
        var ar = ia // BK
        var ac = ia % BK
        var a_src = (Int(block_idx.y) * BM + ar) * kdim + kb * BK + ac
        As.store(ia, a.load[width=4](a_src))
        var ib = tid * 4
        var br = ib // BN
        var bc = ib % BN
        var b_src = (kb * BK + br) * n + Int(block_idx.x) * BN + bc
        Bs.store(ib, b.load[width=4](b_src))
        barrier()
        for kk in range(BK):
            var bv = Bs.load[width=TN](kk * BN + tx * TN)
            acc0 += SIMD[DType.float32, TN](As[(ty * TM + 0) * BK + kk]) * bv
            acc1 += SIMD[DType.float32, TN](As[(ty * TM + 1) * BK + kk]) * bv
            acc2 += SIMD[DType.float32, TN](As[(ty * TM + 2) * BK + kk]) * bv
            acc3 += SIMD[DType.float32, TN](As[(ty * TM + 3) * BK + kk]) * bv
        barrier()
    c.store((row0 + 0) * n + col0, acc0)
    c.store((row0 + 1) * n + col0, acc1)
    c.store((row0 + 2) * n + col0, acc2)
    c.store((row0 + 3) * n + col0, acc3)


def gemm_tiled_a(
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int, n: Int, kdim: Int,
):
    """Variant A: 64x128 tile, 4x8 register block (2 Bs vec4 loads + 4 As
    scalars per kk feed 8 vec4 FMAs). Requires M%64, N%128, K%16."""
    var As = stack_allocation[
        64 * BK, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * 128, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * 64 + ty * 4
    var col0 = Int(block_idx.x) * 128 + tx * 8
    var a00 = SIMD[DType.float32, 4](0)
    var a01 = SIMD[DType.float32, 4](0)
    var a10 = SIMD[DType.float32, 4](0)
    var a11 = SIMD[DType.float32, 4](0)
    var a20 = SIMD[DType.float32, 4](0)
    var a21 = SIMD[DType.float32, 4](0)
    var a30 = SIMD[DType.float32, 4](0)
    var a31 = SIMD[DType.float32, 4](0)
    var n_kb = kdim // BK
    for kb in range(n_kb):
        var ia = tid * 4
        var ar = ia // BK
        var ac = ia % BK
        As.store(ia, a.load[width=4]((Int(block_idx.y) * 64 + ar) * kdim + kb * BK + ac))
        var ib = tid * 8
        var br = ib // 128
        var bc = ib % 128
        var b_src = (kb * BK + br) * n + Int(block_idx.x) * 128 + bc
        Bs.store(ib, b.load[width=4](b_src))
        Bs.store(ib + 4, b.load[width=4](b_src + 4))
        barrier()
        for kk in range(BK):
            var bv0 = Bs.load[width=4](kk * 128 + tx * 8)
            var bv1 = Bs.load[width=4](kk * 128 + tx * 8 + 4)
            var x0 = SIMD[DType.float32, 4](As[(ty * 4 + 0) * BK + kk])
            var x1 = SIMD[DType.float32, 4](As[(ty * 4 + 1) * BK + kk])
            var x2 = SIMD[DType.float32, 4](As[(ty * 4 + 2) * BK + kk])
            var x3 = SIMD[DType.float32, 4](As[(ty * 4 + 3) * BK + kk])
            a00 += x0 * bv0
            a01 += x0 * bv1
            a10 += x1 * bv0
            a11 += x1 * bv1
            a20 += x2 * bv0
            a21 += x2 * bv1
            a30 += x3 * bv0
            a31 += x3 * bv1
        barrier()
    c.store((row0 + 0) * n + col0, a00)
    c.store((row0 + 0) * n + col0 + 4, a01)
    c.store((row0 + 1) * n + col0, a10)
    c.store((row0 + 1) * n + col0 + 4, a11)
    c.store((row0 + 2) * n + col0, a20)
    c.store((row0 + 2) * n + col0 + 4, a21)
    c.store((row0 + 3) * n + col0, a30)
    c.store((row0 + 3) * n + col0 + 4, a31)


def gemm_tiled_b(
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int, n: Int, kdim: Int,
):
    """Variant B: 128x128 tile, 8x8 register block (2 Bs loads + 8 As
    scalars per kk feed 16 vec4 FMAs). Requires M%128, N%128, K%16."""
    var As = stack_allocation[
        128 * BK, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * 128, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * 128 + ty * 8
    var col0 = Int(block_idx.x) * 128 + tx * 8
    var acc = InlineArray[SIMD[DType.float32, 4], 16](fill=SIMD[DType.float32, 4](0))
    var n_kb = kdim // BK
    for kb in range(n_kb):
        var ia = tid * 8
        var ar = ia // BK
        var ac = ia % BK
        var a_src = (Int(block_idx.y) * 128 + ar) * kdim + kb * BK + ac
        As.store(ia, a.load[width=4](a_src))
        As.store(ia + 4, a.load[width=4](a_src + 4))
        var ib = tid * 8
        var br = ib // 128
        var bc = ib % 128
        var b_src = (kb * BK + br) * n + Int(block_idx.x) * 128 + bc
        Bs.store(ib, b.load[width=4](b_src))
        Bs.store(ib + 4, b.load[width=4](b_src + 4))
        barrier()
        for kk in range(BK):
            var bv0 = Bs.load[width=4](kk * 128 + tx * 8)
            var bv1 = Bs.load[width=4](kk * 128 + tx * 8 + 4)
            @parameter
            for r in range(8):
                var xv = SIMD[DType.float32, 4](As[(ty * 8 + r) * BK + kk])
                acc[2 * r] += xv * bv0
                acc[2 * r + 1] += xv * bv1
        barrier()
    @parameter
    for r in range(8):
        c.store((row0 + r) * n + col0, acc[2 * r])
        c.store((row0 + r) * n + col0 + 4, acc[2 * r + 1])


def bench(ctx: DeviceContext, m: Int, n: Int, kdim: Int) raises:
    var ab = ctx.enqueue_create_buffer[DType.float32](m * kdim)
    var bb = ctx.enqueue_create_buffer[DType.float32](kdim * n)
    var cb = ctx.enqueue_create_buffer[DType.float32](m * n)
    with ab.map_to_host() as h:
        for i in range(m * kdim):
            h[i] = Float32((i % 37) - 18) * 0.01
    with bb.map_to_host() as h:
        for i in range(kdim * n):
            h[i] = Float32((i % 29) - 14) * 0.01
    var best: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[gemm_tiled](
            ab.unsafe_ptr(), bb.unsafe_ptr(), cb.unsafe_ptr(), m, n, kdim,
            grid_dim=(n // BN, m // BM), block_dim=(16, 16),
        )
        # commit + wait: remap a host-written input (synchronize() is not
        # enough — see file header)
        with ab.map_to_host() as h:
            _ = h[0]
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best:
            best = ms
    # correctness spot-check vs CPU: one full row
    var maxd: Float32 = 0
    with cb.map_to_host() as hc:
        with ab.map_to_host() as ha:
            with bb.map_to_host() as hb:
                for j in range(n):
                    var exp: Float32 = 0
                    for kk in range(kdim):
                        exp += ha[kdim + kk] * hb[kk * n + j]
                    var d = abs(hc[n + j] - exp)
                    if d > maxd:
                        maxd = d
    var gf = Float64(2 * m) * Float64(n) * Float64(kdim) / (best * 1e6)
    print(m, "x", n, "x", kdim, ":", best, "ms ", gf, "GF/s  row1 max|d|:", maxd)


def gemm_tiled_bf16(
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.uint16], MutAnyOrigin],
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int, n: Int, kdim: Int,
):
    """Current 64x64/4x4 kernel with B stored as bf16 (u16<<16 -> f32 on
    the shared-memory fill) — halves the dominant B device traffic.
    EXACT for the DiT checkpoints (their weights ARE bf16)."""
    var As = stack_allocation[
        BM * BK, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * BN, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * BM + ty * TM
    var col0 = Int(block_idx.x) * BN + tx * TN
    var acc0 = SIMD[DType.float32, TN](0)
    var acc1 = SIMD[DType.float32, TN](0)
    var acc2 = SIMD[DType.float32, TN](0)
    var acc3 = SIMD[DType.float32, TN](0)
    var n_kb = kdim // BK
    for kb in range(n_kb):
        var ia = tid * 4
        var ar = ia // BK
        var ac = ia % BK
        var a_src = (Int(block_idx.y) * BM + ar) * kdim + kb * BK + ac
        As.store(ia, a.load[width=4](a_src))
        var ib = tid * 4
        var br = ib // BN
        var bc = ib % BN
        var b_src = (kb * BK + br) * n + Int(block_idx.x) * BN + bc
        var raw = b.load[width=4](b_src).cast[DType.uint32]() << 16
        Bs.store(ib, SIMD[DType.float32, 4](from_bits=raw))
        barrier()
        for kk in range(BK):
            var bv = Bs.load[width=TN](kk * BN + tx * TN)
            acc0 += SIMD[DType.float32, TN](As[(ty * TM + 0) * BK + kk]) * bv
            acc1 += SIMD[DType.float32, TN](As[(ty * TM + 1) * BK + kk]) * bv
            acc2 += SIMD[DType.float32, TN](As[(ty * TM + 2) * BK + kk]) * bv
            acc3 += SIMD[DType.float32, TN](As[(ty * TM + 3) * BK + kk]) * bv
        barrier()
    c.store((row0 + 0) * n + col0, acc0)
    c.store((row0 + 1) * n + col0, acc1)
    c.store((row0 + 2) * n + col0, acc2)
    c.store((row0 + 3) * n + col0, acc3)


def bench_bf16(ctx: DeviceContext, m: Int, n: Int, kdim: Int) raises:
    var ab = ctx.enqueue_create_buffer[DType.float32](m * kdim)
    var bb = ctx.enqueue_create_buffer[DType.uint16](kdim * n)
    var cb = ctx.enqueue_create_buffer[DType.float32](m * n)
    with ab.map_to_host() as h:
        for i in range(m * kdim):
            h[i] = Float32((i % 37) - 18) * 0.01
    with bb.map_to_host() as h:
        for i in range(kdim * n):
            var f = Float32((i % 29) - 14) * 0.01
            h[i] = UInt16(SIMD[DType.float32, 1](f).to_bits[DType.uint32]() >> 16)
    var best: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[gemm_tiled_bf16](
            ab.unsafe_ptr(), bb.unsafe_ptr(), cb.unsafe_ptr(), m, n, kdim,
            grid_dim=(n // BN, m // BM), block_dim=(16, 16),
        )
        with ab.map_to_host() as h:
            _ = h[0]
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best:
            best = ms
    var gf = Float64(2 * m) * Float64(n) * Float64(kdim) / (best * 1e6)
    print("  bf16-B ", m, "x", n, "x", kdim, ":", best, "ms ", gf, "GF/s")


def bench_variant(ctx: DeviceContext, m: Int, n: Int, kdim: Int, variant: Int) raises:
    """variant 0 = current 64x64/4x4, 1 = A (64x128/4x8), 2 = B (128x128/8x8)."""
    var ab = ctx.enqueue_create_buffer[DType.float32](m * kdim)
    var bb = ctx.enqueue_create_buffer[DType.float32](kdim * n)
    var cb = ctx.enqueue_create_buffer[DType.float32](m * n)
    with ab.map_to_host() as h:
        for i in range(m * kdim):
            h[i] = Float32((i % 37) - 18) * 0.01
    with bb.map_to_host() as h:
        for i in range(kdim * n):
            h[i] = Float32((i % 29) - 14) * 0.01
    var best: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        if variant == 0:
            ctx.enqueue_function[gemm_tiled](
                ab.unsafe_ptr(), bb.unsafe_ptr(), cb.unsafe_ptr(), m, n, kdim,
                grid_dim=(n // BN, m // BM), block_dim=(16, 16),
            )
        elif variant == 1:
            ctx.enqueue_function[gemm_tiled_a](
                ab.unsafe_ptr(), bb.unsafe_ptr(), cb.unsafe_ptr(), m, n, kdim,
                grid_dim=(n // 128, m // 64), block_dim=(16, 16),
            )
        else:
            ctx.enqueue_function[gemm_tiled_b](
                ab.unsafe_ptr(), bb.unsafe_ptr(), cb.unsafe_ptr(), m, n, kdim,
                grid_dim=(n // 128, m // 128), block_dim=(16, 16),
            )
        with ab.map_to_host() as h:
            _ = h[0]
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best:
            best = ms
    var maxd: Float32 = 0
    with cb.map_to_host() as hc:
        with ab.map_to_host() as ha:
            with bb.map_to_host() as hb:
                for j in range(n):
                    var exp: Float32 = 0
                    for kk in range(kdim):
                        exp += ha[kdim + kk] * hb[kk * n + j]
                    var d = abs(hc[n + j] - exp)
                    if d > maxd:
                        maxd = d
    var gf = Float64(2 * m) * Float64(n) * Float64(kdim) / (best * 1e6)
    print(
        "  v", variant, " ", m, "x", n, "x", kdim, ":", best, "ms ",
        gf, "GF/s  row1 max|d|:", maxd,
    )


# ---------------------------------------------------------------------------
# WP19-prober: fp16 i GEMM-kjernen. Hypoteser fra Rigel/M4 (arXiv:
# 2606.12765) + SAR-arbeidet (2604.03585): halvpresisjons-ALU har 2x
# rate og f16->f32-konvertering er ~gratis; kjernen vår er
# shared-memory-/issue-bundet, så gevinsten kan også komme fra halvert
# shared-fotavtrykk (occupancy) og dobbel-buffring (aldri prøvd).
# v3: Bs i f16-shared (cast->f32 i registerlast)
# v4: As OG Bs i f16-shared (A-cast er LOSSY ~5e-4 rel)
# v5: som v4 men MULTIPLIKASJONEN i f16 (acc fortsatt f32) — tester om
#     b2-Metal-backenden emitterer native half-aritmetikk
# v6: som v4 + DOBBEL-BUFFRING (prefetch neste tile mens forrige
#     regnes; én barrier per iterasjon; 2x f16-buffere = samme bytes
#     som én f32)


def gemm_f16_v3(
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.uint16], MutAnyOrigin],
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int, n: Int, kdim: Int,
):
    var As = stack_allocation[
        BM * BK, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * BN, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * BM + ty * TM
    var col0 = Int(block_idx.x) * BN + tx * TN
    var acc0 = SIMD[DType.float32, TN](0)
    var acc1 = SIMD[DType.float32, TN](0)
    var acc2 = SIMD[DType.float32, TN](0)
    var acc3 = SIMD[DType.float32, TN](0)
    var n_kb = kdim // BK
    for kb in range(n_kb):
        var ia = tid * 4
        var ar = ia // BK
        var ac = ia % BK
        var a_src = (Int(block_idx.y) * BM + ar) * kdim + kb * BK + ac
        As.store(ia, a.load[width=4](a_src))
        var ib = tid * 4
        var br = ib // BN
        var bc = ib % BN
        var b_src = (kb * BK + br) * n + Int(block_idx.x) * BN + bc
        var raw = b.load[width=4](b_src)
        Bs.store(ib, SIMD[DType.float16, 4](from_bits=raw))
        barrier()
        for kk in range(BK):
            var bv = Bs.load[width=TN](kk * BN + tx * TN).cast[DType.float32]()
            acc0 += SIMD[DType.float32, TN](As[(ty * TM + 0) * BK + kk]) * bv
            acc1 += SIMD[DType.float32, TN](As[(ty * TM + 1) * BK + kk]) * bv
            acc2 += SIMD[DType.float32, TN](As[(ty * TM + 2) * BK + kk]) * bv
            acc3 += SIMD[DType.float32, TN](As[(ty * TM + 3) * BK + kk]) * bv
        barrier()
    c.store((row0 + 0) * n + col0, acc0)
    c.store((row0 + 1) * n + col0, acc1)
    c.store((row0 + 2) * n + col0, acc2)
    c.store((row0 + 3) * n + col0, acc3)


def gemm_f16_v4(
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.uint16], MutAnyOrigin],
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int, n: Int, kdim: Int,
):
    var As = stack_allocation[
        BM * BK, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * BN, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * BM + ty * TM
    var col0 = Int(block_idx.x) * BN + tx * TN
    var acc0 = SIMD[DType.float32, TN](0)
    var acc1 = SIMD[DType.float32, TN](0)
    var acc2 = SIMD[DType.float32, TN](0)
    var acc3 = SIMD[DType.float32, TN](0)
    var n_kb = kdim // BK
    for kb in range(n_kb):
        var ia = tid * 4
        var ar = ia // BK
        var ac = ia % BK
        var a_src = (Int(block_idx.y) * BM + ar) * kdim + kb * BK + ac
        As.store(ia, a.load[width=4](a_src).cast[DType.float16]())
        var ib = tid * 4
        var br = ib // BN
        var bc = ib % BN
        var b_src = (kb * BK + br) * n + Int(block_idx.x) * BN + bc
        var raw = b.load[width=4](b_src)
        Bs.store(ib, SIMD[DType.float16, 4](from_bits=raw))
        barrier()
        for kk in range(BK):
            var bv = Bs.load[width=TN](kk * BN + tx * TN).cast[DType.float32]()
            acc0 += SIMD[DType.float32, TN](
                As[(ty * TM + 0) * BK + kk].cast[DType.float32]()
            ) * bv
            acc1 += SIMD[DType.float32, TN](
                As[(ty * TM + 1) * BK + kk].cast[DType.float32]()
            ) * bv
            acc2 += SIMD[DType.float32, TN](
                As[(ty * TM + 2) * BK + kk].cast[DType.float32]()
            ) * bv
            acc3 += SIMD[DType.float32, TN](
                As[(ty * TM + 3) * BK + kk].cast[DType.float32]()
            ) * bv
        barrier()
    c.store((row0 + 0) * n + col0, acc0)
    c.store((row0 + 1) * n + col0, acc1)
    c.store((row0 + 2) * n + col0, acc2)
    c.store((row0 + 3) * n + col0, acc3)


def gemm_f16_v5(
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.uint16], MutAnyOrigin],
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int, n: Int, kdim: Int,
):
    var As = stack_allocation[
        BM * BK, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * BN, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * BM + ty * TM
    var col0 = Int(block_idx.x) * BN + tx * TN
    var acc0 = SIMD[DType.float32, TN](0)
    var acc1 = SIMD[DType.float32, TN](0)
    var acc2 = SIMD[DType.float32, TN](0)
    var acc3 = SIMD[DType.float32, TN](0)
    var n_kb = kdim // BK
    for kb in range(n_kb):
        var ia = tid * 4
        var ar = ia // BK
        var ac = ia % BK
        var a_src = (Int(block_idx.y) * BM + ar) * kdim + kb * BK + ac
        As.store(ia, a.load[width=4](a_src).cast[DType.float16]())
        var ib = tid * 4
        var br = ib // BN
        var bc = ib % BN
        var b_src = (kb * BK + br) * n + Int(block_idx.x) * BN + bc
        var raw = b.load[width=4](b_src)
        Bs.store(ib, SIMD[DType.float16, 4](from_bits=raw))
        barrier()
        for kk in range(BK):
            # halv-multiplikasjon, f32-akkumulering
            var bv = Bs.load[width=TN](kk * BN + tx * TN)
            acc0 += (SIMD[DType.float16, TN](As[(ty * TM + 0) * BK + kk]) * bv).cast[DType.float32]()
            acc1 += (SIMD[DType.float16, TN](As[(ty * TM + 1) * BK + kk]) * bv).cast[DType.float32]()
            acc2 += (SIMD[DType.float16, TN](As[(ty * TM + 2) * BK + kk]) * bv).cast[DType.float32]()
            acc3 += (SIMD[DType.float16, TN](As[(ty * TM + 3) * BK + kk]) * bv).cast[DType.float32]()
        barrier()
    c.store((row0 + 0) * n + col0, acc0)
    c.store((row0 + 1) * n + col0, acc1)
    c.store((row0 + 2) * n + col0, acc2)
    c.store((row0 + 3) * n + col0, acc3)


def gemm_f16_v6(
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.uint16], MutAnyOrigin],
    c: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    m: Int, n: Int, kdim: Int,
):
    """v4 + dobbel-buffring: prefetch tile kb+1 mens kb regnes; én
    barrier per iterasjon (to f16-buffere = samme bytes som én f32)."""
    var As = stack_allocation[
        2 * BM * BK, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        2 * BK * BN, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * BM + ty * TM
    var col0 = Int(block_idx.x) * BN + tx * TN
    var acc0 = SIMD[DType.float32, TN](0)
    var acc1 = SIMD[DType.float32, TN](0)
    var acc2 = SIMD[DType.float32, TN](0)
    var acc3 = SIMD[DType.float32, TN](0)
    var n_kb = kdim // BK
    var ia = tid * 4
    var ar = ia // BK
    var ac = ia % BK
    var ib = tid * 4
    var br = ib // BN
    var bc = ib % BN
    # prime buffer 0
    var a_src0 = (Int(block_idx.y) * BM + ar) * kdim + ac
    As.store(ia, a.load[width=4](a_src0).cast[DType.float16]())
    var b_src0 = br * n + Int(block_idx.x) * BN + bc
    Bs.store(ib, SIMD[DType.float16, 4](from_bits=b.load[width=4](b_src0)))
    barrier()
    for kb in range(n_kb):
        var cur = (kb % 2) * BM * BK
        var curb = (kb % 2) * BK * BN
        if kb + 1 < n_kb:
            var nxt = ((kb + 1) % 2) * BM * BK
            var nxtb = ((kb + 1) % 2) * BK * BN
            var a_src = (Int(block_idx.y) * BM + ar) * kdim + (kb + 1) * BK + ac
            As.store(nxt + ia, a.load[width=4](a_src).cast[DType.float16]())
            var b_src = ((kb + 1) * BK + br) * n + Int(block_idx.x) * BN + bc
            Bs.store(nxtb + ib, SIMD[DType.float16, 4](from_bits=b.load[width=4](b_src)))
        for kk in range(BK):
            var bv = Bs.load[width=TN](curb + kk * BN + tx * TN).cast[DType.float32]()
            acc0 += SIMD[DType.float32, TN](
                As[cur + (ty * TM + 0) * BK + kk].cast[DType.float32]()
            ) * bv
            acc1 += SIMD[DType.float32, TN](
                As[cur + (ty * TM + 1) * BK + kk].cast[DType.float32]()
            ) * bv
            acc2 += SIMD[DType.float32, TN](
                As[cur + (ty * TM + 2) * BK + kk].cast[DType.float32]()
            ) * bv
            acc3 += SIMD[DType.float32, TN](
                As[cur + (ty * TM + 3) * BK + kk].cast[DType.float32]()
            ) * bv
        barrier()
    c.store((row0 + 0) * n + col0, acc0)
    c.store((row0 + 1) * n + col0, acc1)
    c.store((row0 + 2) * n + col0, acc2)
    c.store((row0 + 3) * n + col0, acc3)


def bench_wp19(ctx: DeviceContext, m: Int, n: Int, kdim: Int, variant: Int) raises:
    """WP19 fp16-prober: B (og evt. A) i f16; korrekthetssjekk mot CPU-
    referanse regnet på de f16-RUNDEDE verdiene (så v3 skal være ~eksakt
    og v4/v5 vise ren A-cast-effekt)."""
    var ab = ctx.enqueue_create_buffer[DType.float32](m * kdim)
    var bb = ctx.enqueue_create_buffer[DType.uint16](kdim * n)
    var cb = ctx.enqueue_create_buffer[DType.float32](m * n)
    with ab.map_to_host() as h:
        for i in range(m * kdim):
            h[i] = Float32((i % 37) - 18) * 0.01
    with bb.map_to_host() as h:
        for i in range(kdim * n):
            var f = Float32((i % 29) - 14) * 0.01
            h[i] = UInt16(
                SIMD[DType.float32, 1](f).cast[DType.float16]().to_bits[DType.uint16]()
            )
    var best: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        if variant == 3:
            ctx.enqueue_function[gemm_f16_v3](
                ab.unsafe_ptr(), bb.unsafe_ptr(), cb.unsafe_ptr(), m, n, kdim,
                grid_dim=(n // BN, m // BM), block_dim=(16, 16),
            )
        elif variant == 4:
            ctx.enqueue_function[gemm_f16_v4](
                ab.unsafe_ptr(), bb.unsafe_ptr(), cb.unsafe_ptr(), m, n, kdim,
                grid_dim=(n // BN, m // BM), block_dim=(16, 16),
            )
        elif variant == 5:
            ctx.enqueue_function[gemm_f16_v5](
                ab.unsafe_ptr(), bb.unsafe_ptr(), cb.unsafe_ptr(), m, n, kdim,
                grid_dim=(n // BN, m // BM), block_dim=(16, 16),
            )
        else:
            ctx.enqueue_function[gemm_f16_v6](
                ab.unsafe_ptr(), bb.unsafe_ptr(), cb.unsafe_ptr(), m, n, kdim,
                grid_dim=(n // BN, m // BM), block_dim=(16, 16),
            )
        with ab.map_to_host() as h:
            _ = h[0]
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best:
            best = ms
    # korrekthet: rad 1 mot CPU på f16-rundede B-verdier
    var maxd: Float32 = 0
    with cb.map_to_host() as hc:
        with ab.map_to_host() as ha:
            with bb.map_to_host() as hb:
                for j in range(n):
                    var exp: Float32 = 0
                    for kk in range(kdim):
                        var bw = SIMD[DType.float16, 1](
                            from_bits=SIMD[DType.uint16, 1](hb[kk * n + j])
                        ).cast[DType.float32]()
                        exp += ha[kdim + kk] * bw[0]
                    var d = abs(hc[n + j] - exp)
                    if d > maxd:
                        maxd = d
    var gf = Float64(2 * m) * Float64(n) * Float64(kdim) / (best * 1e6)
    print(
        "  wp19-v", variant, " ", m, "x", n, "x", kdim, ":", best, "ms ",
        gf, "GF/s  row1 max|d|:", maxd,
    )


def main() raises:
    var ctx = DeviceContext()
    print("tiled Metal GEMM (min of 3, enqueue -> flush-map wait):")
    bench(ctx, 4096, 1024, 1024)
    bench(ctx, 4096, 3072, 1024)
    bench(ctx, 4096, 4096, 1024)
    bench(ctx, 4096, 1024, 4096)
    bench(ctx, 1024, 1024, 1024)
    print("register-block variants at the REAL DiT shapes (C=1536):")
    for shape in [(4096, 4608, 1536), (4096, 8192, 1536), (4096, 1536, 8192), (4096, 1536, 1536)]:
        for v in range(3):
            bench_variant(ctx, shape[0], shape[1], shape[2], v)
    print("bf16-stored B (weight traffic halved; exact for bf16 ckpts):")
    bench_bf16(ctx, 4096, 4608, 1536)
    bench_bf16(ctx, 4096, 8192, 1536)
    bench_bf16(ctx, 4096, 1536, 8192)
    bench_bf16(ctx, 4096, 1536, 1536)
    print("WP19 fp16-prober (v3 f16-B-shared, v4 f16-A+B, v5 f16-mult, v6 +dbuf):")
    for shape in [(4096, 4608, 1536), (4096, 8192, 1536), (4096, 1536, 8192), (4096, 1536, 1536)]:
        for v in range(3, 7):
            bench_wp19(ctx, shape[0], shape[1], shape[2], v)
