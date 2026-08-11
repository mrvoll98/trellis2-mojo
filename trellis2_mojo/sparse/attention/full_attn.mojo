# Mojo port of the attention math in
#   modules/sparse/attention/full_attn.py  (block-diagonal varlen SDPA)
#   modules/attention/full_attn.py         (dense SDPA)
#
# Everything reduces to one kernel: per-segment softmax(q k^T / sqrt(d)) v,
# where dense batches are just segments of uniform length. This is the
# correctness-first naive kernel (see master plan: performance work is a
# separate, later package).
#
# Ported entry points (the ones the inference models use):
#   sparse: qkv-packed self / q + kv-packed / q + dense kv / q, k, v
#   dense:  qkv-packed / q + kv-packed / q, k, v
# Not ported: dense-q + sparse-kv variants (unused by the models).

from max.algorithm import parallelize
from std.math import exp

from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.sparse.basic import SparseTensor, VarLenTensor

comptime F32 = DType.float32


def uniform_offsets(n: Int, l: Int) raises -> List[Int]:
    var offsets: List[Int] = [0]
    for i in range(n):
        offsets.append((i + 1) * l)
    return offsets^


def varlen_sdpa(
    q: Tensor[F32],
    k: Tensor[F32],
    v: Tensor[F32],
    q_offsets: List[Int],
    kv_offsets: List[Int],
) raises -> Tensor[F32]:
    """Block-diagonal attention. q [Tq, H, Ci], k [Tkv, H, Ci],
    v [Tkv, H, Co] -> [Tq, H, Co]. Segment b of q attends to segment b of
    k/v. Softmax uses stable max-subtraction.

    Hot loops are SIMD over the channel dim (WP10): qk as vectorized dot,
    av as per-key axpy into the out row. Work is parallelized over
    (segment, head) — items write disjoint out regions and get their own
    scores scratch, so results are bit-identical to the serial path
    (which small inputs take to skip thread-spawn overhead). Long q
    segments are further split into chunks of QC rows per work item so
    the item count exceeds the core count even at few (segment, head)
    pairs — every q row computes exactly the same thing regardless of
    which item owns it. Within an item, q rows are processed in tiles of
    TQ with the key loop outermost, so each k/v row is reused TQ times
    while hot in L1. Register tiling (pass 4): qk runs KU keys x TU q
    rows per block with independent accumulator chains, and av
    accumulates each out row in registers across all keys with the
    denominator division fused into the store — per-(qi, kj) math and
    per-lane accumulation order are unchanged, so the EXACT path stays
    bit-identical to the single-pair code.

    Segments with kv_len >= FLASH_KV take a flash path instead (pass 8):
    online softmax over KB-sized kv blocks with the scores tile and
    accumulators in L1 — deterministic but NOT bit-identical to the
    exact path (the running max rescales the accumulator; the denom sum
    is blockwise). Every parity-test shape is below the threshold; the
    flash path is verified against real weights by test-cond and
    test-real."""
    comptime W = 8
    comptime TQ = 32
    comptime KU = 4
    comptime TU = 4
    comptime QC = 256
    comptime FLASH_KV = 1024
    comptime KB = 128
    var h = q.shape[1]
    var ci = q.shape[2]
    var co = v.shape[2]
    if len(q_offsets) != len(kv_offsets):
        raise Error("varlen_sdpa: segment count mismatch")
    if k.shape[1] != h or v.shape[1] != h or k.shape[2] != ci:
        raise Error("varlen_sdpa: head/channel mismatch")
    var scale = Float32(1.0 / Float64(ci) ** 0.5)
    var out_shape: List[Int] = [q.shape[0], h, co]
    var out = Tensor[F32](out_shape)
    var n_seg = len(q_offsets) - 1
    var max_kv = 0
    for seg in range(n_seg):
        var l = kv_offsets[seg + 1] - kv_offsets[seg]
        if l > max_kv:
            max_kv = l
    # work items: (segment, head, q-chunk); chunking long q segments keeps
    # the item count above the core count for load balance
    var item_seg = List[Int]()
    var item_head = List[Int]()
    var item_q0 = List[Int]()
    var item_q1 = List[Int]()
    for seg in range(n_seg):
        var pos = q_offsets[seg]
        var q_end = q_offsets[seg + 1]
        while pos < q_end:
            var hi = min(pos + QC, q_end)
            for head in range(h):
                item_seg.append(seg)
                item_head.append(head)
                item_q0.append(pos)
                item_q1.append(hi)
            pos = hi
    var n_work = len(item_seg)
    var qp = q.data.unsafe_ptr()
    var kp = k.data.unsafe_ptr()
    var vp = v.data.unsafe_ptr()
    var op = out.data.unsafe_ptr()
    var kop = kv_offsets.unsafe_ptr()
    var isp = item_seg.unsafe_ptr()
    var ihp = item_head.unsafe_ptr()
    var iq0p = item_q0.unsafe_ptr()
    var iq1p = item_q1.unsafe_ptr()

    @parameter
    def work(w: Int):
        var seg = isp[unsafe_offset=w]
        var head = ihp[unsafe_offset=w]
        var kv_start = kop[unsafe_offset=seg]
        var kv_len = kop[unsafe_offset=seg + 1] - kv_start
        if kv_len >= FLASH_KV:
            var hco_f = h * co
            # flash path (pass 8): online softmax over kv blocks of KB —
            # the scores tile lives in L1 instead of a [TQ, kv] buffer
            # (whose write + 3 read passes dominated the kernel at 4096
            # kv), and each v row is loaded once per TILE of TQ q rows
            # instead of once per q row. Numerics: the same
            # max-subtracted softmax, but the running max triggers
            # exp-rescales of the accumulator and the denom sum is
            # blockwise — deterministic, NOT bit-identical to the exact
            # path (pass-5 GEMM precedent; verified inside the
            # test-cond/test-real tolerances). Only segments this long
            # take the path, so every parity-test shape (< 1024 kv)
            # stays on the bit-exact kernel.
            var accl = List[Float32](unsafe_uninit_length=TQ * co)
            var mrow = List[Float32](unsafe_uninit_length=TQ)
            var denl = List[Float32](unsafe_uninit_length=TQ)
            var stl = List[Float32](unsafe_uninit_length=TQ * KB)
            var ap = accl.unsafe_ptr()
            var mp = mrow.unsafe_ptr()
            var dp = denl.unsafe_ptr()
            var stp = stl.unsafe_ptr()
            var fq0 = iq0p[unsafe_offset=w]
            var fq_hi = iq1p[unsafe_offset=w]
            while fq0 < fq_hi:
                var tq = min(TQ, fq_hi - fq0)
                for t in range(tq):
                    mp[unsafe_offset=t] = Float32(-3.4e38)
                    dp[unsafe_offset=t] = 0
                var zi = 0
                var zv = SIMD[F32, W](0)
                while zi + W <= tq * co:
                    ap.unsafe_store(zi, zv)
                    zi += W
                while zi < tq * co:
                    ap[unsafe_offset=zi] = 0
                    zi += 1
                var kj = 0
                while kj < kv_len:
                    var kb = min(KB, kv_len - kj)
                    # scores tile [tq, kb] (stride KB): same 4x4 register
                    # blocks as the exact path (16 FMA chains per 8 loads)
                    var hci2 = h * ci
                    var kjj = 0
                    while kjj < kb:
                        var ku = min(4, kb - kjj)
                        var k_base0 = ((kv_start + kj + kjj) * h + head) * ci
                        var t = 0
                        while t < tq:
                            var tu = min(4, tq - t)
                            if ku == 4 and tu == 4:
                                var k_base1 = k_base0 + hci2
                                var k_base2 = k_base1 + hci2
                                var k_base3 = k_base2 + hci2
                                var q_base0 = ((fq0 + t) * h + head) * ci
                                var q_base1 = q_base0 + hci2
                                var q_base2 = q_base1 + hci2
                                var q_base3 = q_base2 + hci2
                                var a00 = SIMD[F32, W](0)
                                var a01 = SIMD[F32, W](0)
                                var a02 = SIMD[F32, W](0)
                                var a03 = SIMD[F32, W](0)
                                var a10 = SIMD[F32, W](0)
                                var a11 = SIMD[F32, W](0)
                                var a12 = SIMD[F32, W](0)
                                var a13 = SIMD[F32, W](0)
                                var a20 = SIMD[F32, W](0)
                                var a21 = SIMD[F32, W](0)
                                var a22 = SIMD[F32, W](0)
                                var a23 = SIMD[F32, W](0)
                                var a30 = SIMD[F32, W](0)
                                var a31 = SIMD[F32, W](0)
                                var a32 = SIMD[F32, W](0)
                                var a33 = SIMD[F32, W](0)
                                var d = 0
                                while d + W <= ci:
                                    var kv0 = kp.unsafe_load[width=W](k_base0 + d)
                                    var kv1 = kp.unsafe_load[width=W](k_base1 + d)
                                    var kv2 = kp.unsafe_load[width=W](k_base2 + d)
                                    var kv3 = kp.unsafe_load[width=W](k_base3 + d)
                                    var qv0 = qp.unsafe_load[width=W](q_base0 + d)
                                    var qv1 = qp.unsafe_load[width=W](q_base1 + d)
                                    var qv2 = qp.unsafe_load[width=W](q_base2 + d)
                                    var qv3 = qp.unsafe_load[width=W](q_base3 + d)
                                    a00 += qv0 * kv0
                                    a01 += qv0 * kv1
                                    a02 += qv0 * kv2
                                    a03 += qv0 * kv3
                                    a10 += qv1 * kv0
                                    a11 += qv1 * kv1
                                    a12 += qv1 * kv2
                                    a13 += qv1 * kv3
                                    a20 += qv2 * kv0
                                    a21 += qv2 * kv1
                                    a22 += qv2 * kv2
                                    a23 += qv2 * kv3
                                    a30 += qv3 * kv0
                                    a31 += qv3 * kv1
                                    a32 += qv3 * kv2
                                    a33 += qv3 * kv3
                                    d += W
                                var s00 = a00.reduce_add()
                                var s01 = a01.reduce_add()
                                var s02 = a02.reduce_add()
                                var s03 = a03.reduce_add()
                                var s10 = a10.reduce_add()
                                var s11 = a11.reduce_add()
                                var s12 = a12.reduce_add()
                                var s13 = a13.reduce_add()
                                var s20 = a20.reduce_add()
                                var s21 = a21.reduce_add()
                                var s22 = a22.reduce_add()
                                var s23 = a23.reduce_add()
                                var s30 = a30.reduce_add()
                                var s31 = a31.reduce_add()
                                var s32 = a32.reduce_add()
                                var s33 = a33.reduce_add()
                                while d < ci:
                                    var k0d = kp[unsafe_offset=k_base0 + d]
                                    var k1d = kp[unsafe_offset=k_base1 + d]
                                    var k2d = kp[unsafe_offset=k_base2 + d]
                                    var k3d = kp[unsafe_offset=k_base3 + d]
                                    s00 += qp[unsafe_offset=q_base0 + d] * k0d
                                    s01 += qp[unsafe_offset=q_base0 + d] * k1d
                                    s02 += qp[unsafe_offset=q_base0 + d] * k2d
                                    s03 += qp[unsafe_offset=q_base0 + d] * k3d
                                    s10 += qp[unsafe_offset=q_base1 + d] * k0d
                                    s11 += qp[unsafe_offset=q_base1 + d] * k1d
                                    s12 += qp[unsafe_offset=q_base1 + d] * k2d
                                    s13 += qp[unsafe_offset=q_base1 + d] * k3d
                                    s20 += qp[unsafe_offset=q_base2 + d] * k0d
                                    s21 += qp[unsafe_offset=q_base2 + d] * k1d
                                    s22 += qp[unsafe_offset=q_base2 + d] * k2d
                                    s23 += qp[unsafe_offset=q_base2 + d] * k3d
                                    s30 += qp[unsafe_offset=q_base3 + d] * k0d
                                    s31 += qp[unsafe_offset=q_base3 + d] * k1d
                                    s32 += qp[unsafe_offset=q_base3 + d] * k2d
                                    s33 += qp[unsafe_offset=q_base3 + d] * k3d
                                    d += 1
                                var sb0 = t * KB + kjj
                                stp[unsafe_offset=sb0] = s00 * scale
                                stp[unsafe_offset=sb0 + 1] = s01 * scale
                                stp[unsafe_offset=sb0 + 2] = s02 * scale
                                stp[unsafe_offset=sb0 + 3] = s03 * scale
                                stp[unsafe_offset=sb0 + KB] = s10 * scale
                                stp[unsafe_offset=sb0 + KB + 1] = s11 * scale
                                stp[unsafe_offset=sb0 + KB + 2] = s12 * scale
                                stp[unsafe_offset=sb0 + KB + 3] = s13 * scale
                                stp[unsafe_offset=sb0 + 2 * KB] = s20 * scale
                                stp[unsafe_offset=sb0 + 2 * KB + 1] = s21 * scale
                                stp[unsafe_offset=sb0 + 2 * KB + 2] = s22 * scale
                                stp[unsafe_offset=sb0 + 2 * KB + 3] = s23 * scale
                                stp[unsafe_offset=sb0 + 3 * KB] = s30 * scale
                                stp[unsafe_offset=sb0 + 3 * KB + 1] = s31 * scale
                                stp[unsafe_offset=sb0 + 3 * KB + 2] = s32 * scale
                                stp[unsafe_offset=sb0 + 3 * KB + 3] = s33 * scale
                            else:
                                for uu in range(ku):
                                    var k_base = k_base0 + uu * hci2
                                    for tt in range(t, t + tu):
                                        var q_base = ((fq0 + tt) * h + head) * ci
                                        var accv = SIMD[F32, W](0)
                                        var d = 0
                                        while d + W <= ci:
                                            accv += qp.unsafe_load[width=W](q_base + d) * kp.unsafe_load[width=W](k_base + d)
                                            d += W
                                        var acc = accv.reduce_add()
                                        while d < ci:
                                            acc += qp[unsafe_offset=q_base + d] * kp[unsafe_offset=k_base + d]
                                            d += 1
                                        stp[unsafe_offset=tt * KB + kjj + uu] = acc * scale
                            t += tu
                        kjj += ku
                    # per row: block max -> rescale accumulator -> exp ->
                    # denom
                    for t in range(tq):
                        var sb = t * KB
                        var bm = Float32(-3.4e38)
                        var u = 0
                        if kb >= W:
                            var bmv = SIMD[F32, W](-3.4e38)
                            while u + W <= kb:
                                bmv = max(bmv, stp.unsafe_load[width=W](sb + u))
                                u += W
                            bm = bmv.reduce_max()
                        while u < kb:
                            if stp[unsafe_offset=sb + u] > bm:
                                bm = stp[unsafe_offset=sb + u]
                            u += 1
                        if bm > mp[unsafe_offset=t]:
                            var r = exp(mp[unsafe_offset=t] - bm)
                            dp[unsafe_offset=t] *= r
                            var rv = SIMD[F32, W](r)
                            var ab = t * co
                            var d2 = 0
                            while d2 + W <= co:
                                ap.unsafe_store(ab + d2, ap.unsafe_load[width=W](ab + d2) * rv)
                                d2 += W
                            while d2 < co:
                                ap[unsafe_offset=ab + d2] *= r
                                d2 += 1
                            mp[unsafe_offset=t] = bm
                        var mv2 = SIMD[F32, W](mp[unsafe_offset=t])
                        var sv2 = SIMD[F32, W](0)
                        u = 0
                        while u + W <= kb:
                            var e = exp(stp.unsafe_load[width=W](sb + u) - mv2)
                            stp.unsafe_store(sb + u, e)
                            sv2 += e
                            u += W
                        var dsum = sv2.reduce_add()
                        while u < kb:
                            var e = exp(stp[unsafe_offset=sb + u] - mp[unsafe_offset=t])
                            stp[unsafe_offset=sb + u] = e
                            dsum += e
                            u += 1
                        dp[unsafe_offset=t] += dsum
                    # av: acc rows held in registers across the block's
                    # keys (the v block is ~32 KB and stays in L1 across
                    # the tq rows); 2 q rows per block share every v load
                    # (16 FMAs per 10 loads), 8W fast path for co=64 with
                    # a chunked single-row ladder for tails
                    var v_base00 = ((kv_start + kj) * h + head) * co
                    var tp = 0
                    while tp + 2 <= tq:
                        if co == 8 * W:
                            var sb2a = tp * KB
                            var sb2b = (tp + 1) * KB
                            var aba = tp * co
                            var abb = (tp + 1) * co
                            var c0 = ap.unsafe_load[width=W](aba)
                            var c1 = ap.unsafe_load[width=W](aba + W)
                            var c2 = ap.unsafe_load[width=W](aba + 2 * W)
                            var c3 = ap.unsafe_load[width=W](aba + 3 * W)
                            var c4 = ap.unsafe_load[width=W](aba + 4 * W)
                            var c5 = ap.unsafe_load[width=W](aba + 5 * W)
                            var c6 = ap.unsafe_load[width=W](aba + 6 * W)
                            var c7 = ap.unsafe_load[width=W](aba + 7 * W)
                            var e0 = ap.unsafe_load[width=W](abb)
                            var e1 = ap.unsafe_load[width=W](abb + W)
                            var e2 = ap.unsafe_load[width=W](abb + 2 * W)
                            var e3 = ap.unsafe_load[width=W](abb + 3 * W)
                            var e4 = ap.unsafe_load[width=W](abb + 4 * W)
                            var e5 = ap.unsafe_load[width=W](abb + 5 * W)
                            var e6 = ap.unsafe_load[width=W](abb + 6 * W)
                            var e7 = ap.unsafe_load[width=W](abb + 7 * W)
                            for u in range(kb):
                                var sa = SIMD[F32, W](stp[unsafe_offset=sb2a + u])
                                var sb3 = SIMD[F32, W](stp[unsafe_offset=sb2b + u])
                                var vb = v_base00 + u * hco_f
                                var v0 = vp.unsafe_load[width=W](vb)
                                var v1 = vp.unsafe_load[width=W](vb + W)
                                var v2 = vp.unsafe_load[width=W](vb + 2 * W)
                                var v3 = vp.unsafe_load[width=W](vb + 3 * W)
                                var v4 = vp.unsafe_load[width=W](vb + 4 * W)
                                var v5 = vp.unsafe_load[width=W](vb + 5 * W)
                                var v6 = vp.unsafe_load[width=W](vb + 6 * W)
                                var v7 = vp.unsafe_load[width=W](vb + 7 * W)
                                c0 += v0 * sa
                                c1 += v1 * sa
                                c2 += v2 * sa
                                c3 += v3 * sa
                                c4 += v4 * sa
                                c5 += v5 * sa
                                c6 += v6 * sa
                                c7 += v7 * sa
                                e0 += v0 * sb3
                                e1 += v1 * sb3
                                e2 += v2 * sb3
                                e3 += v3 * sb3
                                e4 += v4 * sb3
                                e5 += v5 * sb3
                                e6 += v6 * sb3
                                e7 += v7 * sb3
                            ap.unsafe_store(aba, c0)
                            ap.unsafe_store(aba + W, c1)
                            ap.unsafe_store(aba + 2 * W, c2)
                            ap.unsafe_store(aba + 3 * W, c3)
                            ap.unsafe_store(aba + 4 * W, c4)
                            ap.unsafe_store(aba + 5 * W, c5)
                            ap.unsafe_store(aba + 6 * W, c6)
                            ap.unsafe_store(aba + 7 * W, c7)
                            ap.unsafe_store(abb, e0)
                            ap.unsafe_store(abb + W, e1)
                            ap.unsafe_store(abb + 2 * W, e2)
                            ap.unsafe_store(abb + 3 * W, e3)
                            ap.unsafe_store(abb + 4 * W, e4)
                            ap.unsafe_store(abb + 5 * W, e5)
                            ap.unsafe_store(abb + 6 * W, e6)
                            ap.unsafe_store(abb + 7 * W, e7)
                            tp += 2
                        else:
                            break
                    while tp < tq:
                        var t = tp
                        var sb2 = t * KB
                        var ab = t * co
                        var v_base0 = v_base00
                        var d3 = 0
                        while d3 + 8 * W <= co:
                            var b0 = ap.unsafe_load[width=W](ab + d3)
                            var b1 = ap.unsafe_load[width=W](ab + d3 + W)
                            var b2 = ap.unsafe_load[width=W](ab + d3 + 2 * W)
                            var b3 = ap.unsafe_load[width=W](ab + d3 + 3 * W)
                            var b4 = ap.unsafe_load[width=W](ab + d3 + 4 * W)
                            var b5 = ap.unsafe_load[width=W](ab + d3 + 5 * W)
                            var b6 = ap.unsafe_load[width=W](ab + d3 + 6 * W)
                            var b7 = ap.unsafe_load[width=W](ab + d3 + 7 * W)
                            for u in range(kb):
                                var sv = SIMD[F32, W](stp[unsafe_offset=sb2 + u])
                                var vb = v_base0 + u * hco_f + d3
                                b0 += vp.unsafe_load[width=W](vb) * sv
                                b1 += vp.unsafe_load[width=W](vb + W) * sv
                                b2 += vp.unsafe_load[width=W](vb + 2 * W) * sv
                                b3 += vp.unsafe_load[width=W](vb + 3 * W) * sv
                                b4 += vp.unsafe_load[width=W](vb + 4 * W) * sv
                                b5 += vp.unsafe_load[width=W](vb + 5 * W) * sv
                                b6 += vp.unsafe_load[width=W](vb + 6 * W) * sv
                                b7 += vp.unsafe_load[width=W](vb + 7 * W) * sv
                            ap.unsafe_store(ab + d3, b0)
                            ap.unsafe_store(ab + d3 + W, b1)
                            ap.unsafe_store(ab + d3 + 2 * W, b2)
                            ap.unsafe_store(ab + d3 + 3 * W, b3)
                            ap.unsafe_store(ab + d3 + 4 * W, b4)
                            ap.unsafe_store(ab + d3 + 5 * W, b5)
                            ap.unsafe_store(ab + d3 + 6 * W, b6)
                            ap.unsafe_store(ab + d3 + 7 * W, b7)
                            d3 += 8 * W
                        while d3 + W <= co:
                            var b0 = ap.unsafe_load[width=W](ab + d3)
                            for u in range(kb):
                                var vb = v_base0 + u * hco_f + d3
                                b0 += vp.unsafe_load[width=W](vb) * SIMD[F32, W](stp[unsafe_offset=sb2 + u])
                            ap.unsafe_store(ab + d3, b0)
                            d3 += W
                        while d3 < co:
                            var acc2: Float32 = ap[unsafe_offset=ab + d3]
                            for u in range(kb):
                                acc2 += vp[unsafe_offset=v_base0 + u * hco_f + d3] * stp[unsafe_offset=sb2 + u]
                            ap[unsafe_offset=ab + d3] = acc2
                            d3 += 1
                        tp += 1
                    kj += kb
                for t in range(tq):
                    var o_base = ((fq0 + t) * h + head) * co
                    var ab = t * co
                    var dv = SIMD[F32, W](dp[unsafe_offset=t])
                    var d4 = 0
                    while d4 + W <= co:
                        op.unsafe_store(o_base + d4, ap.unsafe_load[width=W](ab + d4) / dv)
                        d4 += W
                    while d4 < co:
                        op[unsafe_offset=o_base + d4] = ap[unsafe_offset=ab + d4] / dp[unsafe_offset=t]
                        d4 += 1
                fq0 += tq
            return
        # per-item scratch (pass 8): the old global n_work x TQ x max_kv
        # buffer capped TQ at 8 — bigger q tiles cut the k re-streaming
        # per tile linearly, which is what this kernel is bound by (the
        # per-(seg, head) k slice lives in L2, ~1 MB at 4096 kv)
        # uninit is safe: qk writes every (t < tq, kj < kv_len) position
        # a tile reads, and denoms[t] is written before use (queue item 1)
        var scores = List[Float32](unsafe_uninit_length=TQ * kv_len)
        var denoms = List[Float32](unsafe_uninit_length=TQ)
        var sp = scores.unsafe_ptr()
        var dnp = denoms.unsafe_ptr()
        var sbase0 = 0
        var q_hi = iq1p[unsafe_offset=w]
        var q0 = iq0p[unsafe_offset=w]
        var hci = h * ci
        var hco = h * co
        while q0 < q_hi:
            var tq = min(TQ, q_hi - q0)
            # scores tile: key loop outermost so the k rows serve all tq
            # rows; KU keys x TU q rows per register block -> 8 independent
            # FMA chains sharing the k/q loads
            var kj = 0
            while kj < kv_len:
                var ku = min(KU, kv_len - kj)
                var k_base0 = ((kv_start + kj) * h + head) * ci
                var t = 0
                while t < tq:
                    var tu = min(TU, tq - t)
                    if ku == 4 and tu == 4:
                        # 4 keys x 4 q rows: 16 independent FMA chains per
                        # 8 loads (pass 8; was 2x4 = 8 chains per 6 loads).
                        # Per-(qi, kj) math and d-order unchanged ->
                        # bit-identical.
                        var k_base1 = k_base0 + hci
                        var k_base2 = k_base1 + hci
                        var k_base3 = k_base2 + hci
                        var q_base0 = ((q0 + t) * h + head) * ci
                        var q_base1 = q_base0 + hci
                        var q_base2 = q_base1 + hci
                        var q_base3 = q_base2 + hci
                        var a00 = SIMD[F32, W](0)
                        var a01 = SIMD[F32, W](0)
                        var a02 = SIMD[F32, W](0)
                        var a03 = SIMD[F32, W](0)
                        var a10 = SIMD[F32, W](0)
                        var a11 = SIMD[F32, W](0)
                        var a12 = SIMD[F32, W](0)
                        var a13 = SIMD[F32, W](0)
                        var a20 = SIMD[F32, W](0)
                        var a21 = SIMD[F32, W](0)
                        var a22 = SIMD[F32, W](0)
                        var a23 = SIMD[F32, W](0)
                        var a30 = SIMD[F32, W](0)
                        var a31 = SIMD[F32, W](0)
                        var a32 = SIMD[F32, W](0)
                        var a33 = SIMD[F32, W](0)
                        var d = 0
                        while d + W <= ci:
                            var kv0 = kp.unsafe_load[width=W](k_base0 + d)
                            var kv1 = kp.unsafe_load[width=W](k_base1 + d)
                            var kv2 = kp.unsafe_load[width=W](k_base2 + d)
                            var kv3 = kp.unsafe_load[width=W](k_base3 + d)
                            var qv0 = qp.unsafe_load[width=W](q_base0 + d)
                            var qv1 = qp.unsafe_load[width=W](q_base1 + d)
                            var qv2 = qp.unsafe_load[width=W](q_base2 + d)
                            var qv3 = qp.unsafe_load[width=W](q_base3 + d)
                            a00 += qv0 * kv0
                            a01 += qv0 * kv1
                            a02 += qv0 * kv2
                            a03 += qv0 * kv3
                            a10 += qv1 * kv0
                            a11 += qv1 * kv1
                            a12 += qv1 * kv2
                            a13 += qv1 * kv3
                            a20 += qv2 * kv0
                            a21 += qv2 * kv1
                            a22 += qv2 * kv2
                            a23 += qv2 * kv3
                            a30 += qv3 * kv0
                            a31 += qv3 * kv1
                            a32 += qv3 * kv2
                            a33 += qv3 * kv3
                            d += W
                        var s00 = a00.reduce_add()
                        var s01 = a01.reduce_add()
                        var s02 = a02.reduce_add()
                        var s03 = a03.reduce_add()
                        var s10 = a10.reduce_add()
                        var s11 = a11.reduce_add()
                        var s12 = a12.reduce_add()
                        var s13 = a13.reduce_add()
                        var s20 = a20.reduce_add()
                        var s21 = a21.reduce_add()
                        var s22 = a22.reduce_add()
                        var s23 = a23.reduce_add()
                        var s30 = a30.reduce_add()
                        var s31 = a31.reduce_add()
                        var s32 = a32.reduce_add()
                        var s33 = a33.reduce_add()
                        while d < ci:
                            var k0d = kp[unsafe_offset=k_base0 + d]
                            var k1d = kp[unsafe_offset=k_base1 + d]
                            var k2d = kp[unsafe_offset=k_base2 + d]
                            var k3d = kp[unsafe_offset=k_base3 + d]
                            s00 += qp[unsafe_offset=q_base0 + d] * k0d
                            s01 += qp[unsafe_offset=q_base0 + d] * k1d
                            s02 += qp[unsafe_offset=q_base0 + d] * k2d
                            s03 += qp[unsafe_offset=q_base0 + d] * k3d
                            s10 += qp[unsafe_offset=q_base1 + d] * k0d
                            s11 += qp[unsafe_offset=q_base1 + d] * k1d
                            s12 += qp[unsafe_offset=q_base1 + d] * k2d
                            s13 += qp[unsafe_offset=q_base1 + d] * k3d
                            s20 += qp[unsafe_offset=q_base2 + d] * k0d
                            s21 += qp[unsafe_offset=q_base2 + d] * k1d
                            s22 += qp[unsafe_offset=q_base2 + d] * k2d
                            s23 += qp[unsafe_offset=q_base2 + d] * k3d
                            s30 += qp[unsafe_offset=q_base3 + d] * k0d
                            s31 += qp[unsafe_offset=q_base3 + d] * k1d
                            s32 += qp[unsafe_offset=q_base3 + d] * k2d
                            s33 += qp[unsafe_offset=q_base3 + d] * k3d
                            d += 1
                        var sb0 = sbase0 + t * kv_len + kj
                        sp[unsafe_offset=sb0] = s00 * scale
                        sp[unsafe_offset=sb0 + 1] = s01 * scale
                        sp[unsafe_offset=sb0 + 2] = s02 * scale
                        sp[unsafe_offset=sb0 + 3] = s03 * scale
                        sp[unsafe_offset=sb0 + kv_len] = s10 * scale
                        sp[unsafe_offset=sb0 + kv_len + 1] = s11 * scale
                        sp[unsafe_offset=sb0 + kv_len + 2] = s12 * scale
                        sp[unsafe_offset=sb0 + kv_len + 3] = s13 * scale
                        sp[unsafe_offset=sb0 + 2 * kv_len] = s20 * scale
                        sp[unsafe_offset=sb0 + 2 * kv_len + 1] = s21 * scale
                        sp[unsafe_offset=sb0 + 2 * kv_len + 2] = s22 * scale
                        sp[unsafe_offset=sb0 + 2 * kv_len + 3] = s23 * scale
                        sp[unsafe_offset=sb0 + 3 * kv_len] = s30 * scale
                        sp[unsafe_offset=sb0 + 3 * kv_len + 1] = s31 * scale
                        sp[unsafe_offset=sb0 + 3 * kv_len + 2] = s32 * scale
                        sp[unsafe_offset=sb0 + 3 * kv_len + 3] = s33 * scale
                    else:
                        for u in range(ku):
                            var k_base = k_base0 + u * hci
                            for tt in range(t, t + tu):
                                var q_base = ((q0 + tt) * h + head) * ci
                                var accv = SIMD[F32, W](0)
                                var d = 0
                                while d + W <= ci:
                                    accv += qp.unsafe_load[width=W](q_base + d) * kp.unsafe_load[width=W](k_base + d)
                                    d += W
                                var acc = accv.reduce_add()
                                while d < ci:
                                    acc += qp[unsafe_offset=q_base + d] * kp[unsafe_offset=k_base + d]
                                    d += 1
                                sp[unsafe_offset=sbase0 + tt * kv_len + kj + u] = acc * scale
                    t += tu
                kj += ku
            # softmax per row with stable max-subtraction. SIMD (pass
            # 8): max via lane-max + reduce_max (order-free -> exact), exp
            # W lanes at a time (the vector exp matches the scalar exp
            # element-for-element — same activation-kernel precedent), and
            # the denominator stays a SEQUENTIAL scalar sum over kj ->
            # bit-identical to the scalar loop. The scalar exp was ~half
            # the kernel time at 4096 kv (268M libm-class calls/forward).
            for t in range(tq):
                var sb = sbase0 + t * kv_len
                var m = Float32(-3.4e38)
                var kj3 = 0
                if kv_len >= W:
                    var mv = SIMD[F32, W](-3.4e38)
                    while kj3 + W <= kv_len:
                        mv = max(mv, sp.unsafe_load[width=W](sb + kj3))
                        kj3 += W
                    m = mv.reduce_max()
                while kj3 < kv_len:
                    if sp[unsafe_offset=sb + kj3] > m:
                        m = sp[unsafe_offset=sb + kj3]
                    kj3 += 1
                kj3 = 0
                while kj3 + W <= kv_len:
                    sp.unsafe_store(sb + kj3, exp(sp.unsafe_load[width=W](sb + kj3) - m))
                    kj3 += W
                while kj3 < kv_len:
                    sp[unsafe_offset=sb + kj3] = exp(sp[unsafe_offset=sb + kj3] - m)
                    kj3 += 1
                var denom: Float32 = 0
                for kj4 in range(kv_len):
                    denom += sp[unsafe_offset=sb + kj4]
                dnp[unsafe_offset=t] = denom
            # av: each out row is accumulated in registers across all keys
            # (same per-lane add order over kj as the axpy formulation) and
            # the denominator division is fused into the store
            for t in range(tq):
                var sb = sbase0 + t * kv_len
                var denom = dnp[unsafe_offset=t]
                var o_base = ((q0 + t) * h + head) * co
                var v_base0 = (kv_start * h + head) * co
                var d = 0
                while d + 8 * W <= co:
                    var b0 = SIMD[F32, W](0)
                    var b1 = SIMD[F32, W](0)
                    var b2 = SIMD[F32, W](0)
                    var b3 = SIMD[F32, W](0)
                    var b4 = SIMD[F32, W](0)
                    var b5 = SIMD[F32, W](0)
                    var b6 = SIMD[F32, W](0)
                    var b7 = SIMD[F32, W](0)
                    for kj2 in range(kv_len):
                        var sv = SIMD[F32, W](sp[unsafe_offset=sb + kj2])
                        var vb = v_base0 + kj2 * hco + d
                        b0 += vp.unsafe_load[width=W](vb) * sv
                        b1 += vp.unsafe_load[width=W](vb + W) * sv
                        b2 += vp.unsafe_load[width=W](vb + 2 * W) * sv
                        b3 += vp.unsafe_load[width=W](vb + 3 * W) * sv
                        b4 += vp.unsafe_load[width=W](vb + 4 * W) * sv
                        b5 += vp.unsafe_load[width=W](vb + 5 * W) * sv
                        b6 += vp.unsafe_load[width=W](vb + 6 * W) * sv
                        b7 += vp.unsafe_load[width=W](vb + 7 * W) * sv
                    var dv = SIMD[F32, W](denom)
                    op.unsafe_store(o_base + d, b0 / dv)
                    op.unsafe_store(o_base + d + W, b1 / dv)
                    op.unsafe_store(o_base + d + 2 * W, b2 / dv)
                    op.unsafe_store(o_base + d + 3 * W, b3 / dv)
                    op.unsafe_store(o_base + d + 4 * W, b4 / dv)
                    op.unsafe_store(o_base + d + 5 * W, b5 / dv)
                    op.unsafe_store(o_base + d + 6 * W, b6 / dv)
                    op.unsafe_store(o_base + d + 7 * W, b7 / dv)
                    d += 8 * W
                while d + 4 * W <= co:
                    var b0 = SIMD[F32, W](0)
                    var b1 = SIMD[F32, W](0)
                    var b2 = SIMD[F32, W](0)
                    var b3 = SIMD[F32, W](0)
                    for kj2 in range(kv_len):
                        var sv = SIMD[F32, W](sp[unsafe_offset=sb + kj2])
                        var vb = v_base0 + kj2 * hco + d
                        b0 += vp.unsafe_load[width=W](vb) * sv
                        b1 += vp.unsafe_load[width=W](vb + W) * sv
                        b2 += vp.unsafe_load[width=W](vb + 2 * W) * sv
                        b3 += vp.unsafe_load[width=W](vb + 3 * W) * sv
                    var dv = SIMD[F32, W](denom)
                    op.unsafe_store(o_base + d, b0 / dv)
                    op.unsafe_store(o_base + d + W, b1 / dv)
                    op.unsafe_store(o_base + d + 2 * W, b2 / dv)
                    op.unsafe_store(o_base + d + 3 * W, b3 / dv)
                    d += 4 * W
                while d + W <= co:
                    var b0 = SIMD[F32, W](0)
                    for kj2 in range(kv_len):
                        var vb = v_base0 + kj2 * hco + d
                        b0 += vp.unsafe_load[width=W](vb) * SIMD[F32, W](sp[unsafe_offset=sb + kj2])
                    op.unsafe_store(o_base + d, b0 / SIMD[F32, W](denom))
                    d += W
                while d < co:
                    var acc: Float32 = 0
                    for kj2 in range(kv_len):
                        acc += vp[unsafe_offset=v_base0 + kj2 * hco + d] * sp[unsafe_offset=sb + kj2]
                    op[unsafe_offset=o_base + d] = acc / denom
                    d += 1
            q0 += tq

    # threshold tuned on the WP10 sampler case: spawn/join overhead beats
    # the parallel gain below ~0.5M flops-proxy
    if n_work == 1 or q.shape[0] * max_kv * ci < 1 << 19:
        for w in range(n_work):
            work(w)
    else:
        parallelize[work](n_work)
    return out^


# -- sparse entry points ------------------------------------------------------

def sparse_sdpa_qkv(qkv: SparseTensor[F32]) raises -> SparseTensor[F32]:
    """Self-attention on packed qkv feats [T, 3, H, C]."""
    var parts = qkv.vl.feats.unbind(1)
    var out = varlen_sdpa(parts[0], parts[1], parts[2], qkv.vl.offsets, qkv.vl.offsets)
    return qkv.replace(out^)


def sparse_sdpa_q_kv(q: SparseTensor[F32], kv: SparseTensor[F32]) raises -> SparseTensor[F32]:
    """q feats [Tq, H, C], kv feats [Tkv, 2, H, C] (own coords, same B)."""
    var parts = kv.vl.feats.unbind(1)
    var out = varlen_sdpa(q.vl.feats, parts[0], parts[1], q.vl.offsets, kv.vl.offsets)
    return q.replace(out^)


def sparse_sdpa_q_kv_dense(q: SparseTensor[F32], kv: Tensor[F32]) raises -> SparseTensor[F32]:
    """q sparse [Tq, H, C], kv dense [N, L, 2, H, C] — the cross-attention
    path against dense conditioning (image features)."""
    var n = kv.shape[0]
    var l = kv.shape[1]
    if n != q.batch_size():
        raise Error("sparse_sdpa_q_kv_dense: batch mismatch")
    var parts = kv.flatten_leading(2).unbind(1)  # [N*L, H, C] x2
    var out = varlen_sdpa(q.vl.feats, parts[0], parts[1], q.vl.offsets, uniform_offsets(n, l))
    return q.replace(out^)


def sparse_sdpa_q_k_v(
    q: SparseTensor[F32], k: SparseTensor[F32], v: SparseTensor[F32]
) raises -> SparseTensor[F32]:
    """Separate q [Tq, H, Ci], k [Tkv, H, Ci], v [Tkv, H, Co]; k and v share
    coords."""
    var out = varlen_sdpa(q.vl.feats, k.vl.feats, v.vl.feats, q.vl.offsets, k.vl.offsets)
    return q.replace(out^)


# -- dense entry points -------------------------------------------------------

def dense_sdpa_qkv(qkv: Tensor[F32]) raises -> Tensor[F32]:
    """qkv [N, L, 3, H, C] -> [N, L, H, C]."""
    var n = qkv.shape[0]
    var l = qkv.shape[1]
    var parts = qkv.flatten_leading(2).unbind(1)
    var offsets = uniform_offsets(n, l)
    var out = varlen_sdpa(parts[0], parts[1], parts[2], offsets, offsets)
    var tail: List[Int] = [n, l, out.shape[1], out.shape[2]]
    return Tensor[F32].from_values(tail, out.data)


def dense_sdpa_q_kv(q: Tensor[F32], kv: Tensor[F32]) raises -> Tensor[F32]:
    """q [N, L, H, C], kv [N, Lkv, 2, H, C]."""
    var n = q.shape[0]
    var l = q.shape[1]
    var lkv = kv.shape[1]
    var parts = kv.flatten_leading(2).unbind(1)
    var out = varlen_sdpa(
        q.flatten_leading(2), parts[0], parts[1], uniform_offsets(n, l), uniform_offsets(n, lkv)
    )
    var tail: List[Int] = [n, l, out.shape[1], out.shape[2]]
    return Tensor[F32].from_values(tail, out.data)


def dense_sdpa_q_k_v(q: Tensor[F32], k: Tensor[F32], v: Tensor[F32]) raises -> Tensor[F32]:
    """q [N, L, H, Ci], k [N, Lkv, H, Ci], v [N, Lkv, H, Co]."""
    var n = q.shape[0]
    var l = q.shape[1]
    var lkv = k.shape[1]
    var out = varlen_sdpa(
        q.flatten_leading(2),
        k.flatten_leading(2),
        v.flatten_leading(2),
        uniform_offsets(n, l),
        uniform_offsets(n, lkv),
    )
    var tail: List[Int] = [n, l, out.shape[1], out.shape[2]]
    return Tensor[F32].from_values(tail, out.data)
