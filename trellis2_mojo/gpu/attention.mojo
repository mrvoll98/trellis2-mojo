# WP11 step 3: dense SDPA on the Metal GPU as a GEMM composition with
# device-resident intermediates — the scores matrix (the 1 GB/forward
# traffic that motivated the CPU flash path in pass 8) never leaves the
# GPU:
#
#   qk:      scores[unsafe_offset=h] = (q[unsafe_offset=h] * scale) @ k[h]^T      (gemm_z, batched
#            over heads via grid z; the scale is pre-baked into q during
#            packing so the kernel needs no float scalar)
#   softmax: row max over the VALID columns, p = exp(x - m) in place,
#            0 for padded columns, row sums to a side buffer (the 1/sum
#            is NOT applied on the GPU...)
#   av:      out[h] = p[unsafe_offset=h] @ v[h]                     (gemm_z again)
#   readback (...it is fused into the parallel readback pass on the CPU
#            together with the [T,H,D] re-interleave).
#
# BOTH sides pad to 64 (WP11 step 4): kv gets zero k-columns/v-rows plus
# a softmax mask, and q gets zero rows whose outputs are simply dropped in
# the readback (the whole composition is q-row-independent), so arbitrary
# lengths work on both sides. That makes the single-segment varlen case
# (B=1 sparse slat DiTs, arbitrary token counts) the same code path:
# gpu_varlen_sdpa_single takes the [T, H, D] layout varlen_sdpa uses.
#
# WP11 step 7 adds the fully device-resident SELF-ATTENTION CHAIN
# (gpu_attn_self_chain): qkv-GEMM -> bias + qk-rms + rope (one fused
# kernel over the valid rows) -> head-major pack -> the SDPA composition
# above -> un-pack with 1/sum fused -> out-GEMM, with only x uploaded and
# only the final [T, C] read back. Per-MHA constants (qkv bias + rms
# gammas) ride in ONE device buffer (GpuAttnChain, built at model load)
# so every kernel stays within the 4-binding marshalling law; the sdpa
# scale and sqrt(D) are recomputed in-kernel from the Int d (float
# scalar args are avoided everywhere — step 3 precedent). Scratch use
# mirrors gpu_mlp_forward: linear A holds x then the final GEMM output,
# linear C holds qkv then the re-interleaved attention output (safe: the
# queue executes in order), plus the attention scratches for q/k/v/
# scores/out/sums and the uploaded rope phases.
#
# Numerics: GPU exp and per-element accumulation order differ from BOTH
# CPU paths (exact and flash) -> tolerance parity, never bit-equality
# (pass 8 precedent). All 1.0.0b2-Metal laws from gpu/linear.mojo apply;
# NEW trap found for this step (probes 2026-07-10): the FIRST full
# write->kernel->read cycle in a process delivers corrupt reads at
# 256-byte boundaries regardless of fences or per-kernel warm-up launches
# — GpuContext runs a sacrificial cycle + a VERIFIED self-test at
# creation (gpu/linear.mojo) and falls back to CPU if the second cycle
# misbehaves.

from max.algorithm import parallelize
from std.gpu import thread_idx, block_idx, block_dim
from max.gpu import barrier
from max.gpu.host import DeviceBuffer
from std.memory import AddressSpace
from std.math import exp, sqrt
from std.memory import stack_allocation, unsafe_memcpy

from trellis2_mojo.gpu.context import GpuContext
from trellis2_mojo.gpu.linear import GpuLinear
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32

comptime BM = 64
comptime BN = 64
comptime BK = 16
comptime TM = 4
comptime TN = 4

# q_len at or above this takes the GPU. WP11 step 13: lowered 2048 -> 1024
# after the golden 12-step run landed at 1857 slat tokens and fell back to
# CPU — measured (H12 D128): 1857 self 3.35x, 1280 2.14x, 1024 still 1.81x
# on the GPU. Below 1024 is unmeasured; the floor stays there.
comptime GPU_SDPA_MIN_Q = 1024
comptime GPU_SDPA_MIN_KV = 512
# refuse to grow the scores scratch past this (Z * L * Lpad floats)
comptime GPU_SDPA_MAX_SCORES = 1 << 28


def gemm_z(
    a: Pointer[Scalar[F32], ImmutAnyOrigin],
    b: Pointer[Scalar[F32], ImmutAnyOrigin],
    c: Pointer[Scalar[F32], MutAnyOrigin],
    n_dev: Int32, kdim_dev: Int32, sa_dev: Int32, sb_dev: Int32, sc_dev: Int32,
):
    """C[z] = A[z] @ B[z] for z in grid.z: the tiled GEMM from
    gpu/linear.mojo with per-z base offsets (sa/sb/sc element strides).
    Requires m % 64 == n % 64 == kdim % 16 == 0."""
    var n = Int(n_dev)
    var kdim = Int(kdim_dev)
    var sa = Int(sa_dev)
    var sb = Int(sb_dev)
    var sc = Int(sc_dev)

    var As = stack_allocation[
        BM * BK, Scalar[F32], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * BN, Scalar[F32], address_space = AddressSpace.SHARED
    ]()
    var z = Int(block_idx.z)
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * BM + ty * TM
    var col0 = Int(block_idx.x) * BN + tx * TN
    var acc0 = SIMD[F32, TN](0)
    var acc1 = SIMD[F32, TN](0)
    var acc2 = SIMD[F32, TN](0)
    var acc3 = SIMD[F32, TN](0)
    var n_kb = kdim // BK
    for kb in range(n_kb):
        var ia = tid * 4
        var ar = ia // BK
        var ac = ia % BK
        var a_src = z * sa + (Int(block_idx.y) * BM + ar) * kdim + kb * BK + ac
        As.unsafe_store(ia, a.unsafe_load[width=4](a_src))
        var ib = tid * 4
        var br = ib // BN
        var bc = ib % BN
        var b_src = z * sb + (kb * BK + br) * n + Int(block_idx.x) * BN + bc
        Bs.unsafe_store(ib, b.unsafe_load[width=4](b_src))
        barrier()
        for kk in range(BK):
            var bv = Bs.unsafe_load[width=TN](kk * BN + tx * TN)
            acc0 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 0) * BK + kk]) * bv
            acc1 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 1) * BK + kk]) * bv
            acc2 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 2) * BK + kk]) * bv
            acc3 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 3) * BK + kk]) * bv
        barrier()
    var cb = z * sc
    c.unsafe_store(cb + (row0 + 0) * n + col0, acc0)
    c.unsafe_store(cb + (row0 + 1) * n + col0, acc1)
    c.unsafe_store(cb + (row0 + 2) * n + col0, acc2)
    c.unsafe_store(cb + (row0 + 3) * n + col0, acc3)


def gemm_z_f16sh(
    a: Pointer[Scalar[F32], ImmutAnyOrigin],
    b: Pointer[Scalar[F32], ImmutAnyOrigin],
    c: Pointer[Scalar[F32], MutAnyOrigin],
    n_dev: Int32, kdim_dev: Int32, sa_dev: Int32, sb_dev: Int32, sc_dev: Int32,
):
    """WP19 trinn 2: gemm_z med BEGGE shared-flisene i f16 (kildene er
    f32-aktiveringer — q/k/v/probs — og castes på fyllet; matte og
    akkumulatorer i f32). Samme transformasjon som gemm_tiled_f16sh:
    +32-40 % målt på flis-geometrien (shared-båndbredde-bundet).
    Numerikk: ~5e-4 rel per element inn i qk-logitene — opt-in bak
    TRELLIS2_GPU_F16 sammen med vekt-GEMM-ene."""
    var n = Int(n_dev)
    var kdim = Int(kdim_dev)
    var sa = Int(sa_dev)
    var sb = Int(sb_dev)
    var sc = Int(sc_dev)

    var As = stack_allocation[
        BM * BK, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * BN, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var z = Int(block_idx.z)
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * BM + ty * TM
    var col0 = Int(block_idx.x) * BN + tx * TN
    var acc0 = SIMD[F32, TN](0)
    var acc1 = SIMD[F32, TN](0)
    var acc2 = SIMD[F32, TN](0)
    var acc3 = SIMD[F32, TN](0)
    var n_kb = kdim // BK
    for kb in range(n_kb):
        var ia = tid * 4
        var ar = ia // BK
        var ac = ia % BK
        var a_src = z * sa + (Int(block_idx.y) * BM + ar) * kdim + kb * BK + ac
        As.unsafe_store(ia, a.unsafe_load[width=4](a_src).cast[DType.float16]())
        var ib = tid * 4
        var br = ib // BN
        var bc = ib % BN
        var b_src = z * sb + (kb * BK + br) * n + Int(block_idx.x) * BN + bc
        Bs.unsafe_store(ib, b.unsafe_load[width=4](b_src).cast[DType.float16]())
        barrier()
        for kk in range(BK):
            var bv = Bs.unsafe_load[width=TN](kk * BN + tx * TN).cast[F32]()
            acc0 += SIMD[F32, TN](
                As[unsafe_offset=(ty * TM + 0) * BK + kk].cast[F32]()
            ) * bv
            acc1 += SIMD[F32, TN](
                As[unsafe_offset=(ty * TM + 1) * BK + kk].cast[F32]()
            ) * bv
            acc2 += SIMD[F32, TN](
                As[unsafe_offset=(ty * TM + 2) * BK + kk].cast[F32]()
            ) * bv
            acc3 += SIMD[F32, TN](
                As[unsafe_offset=(ty * TM + 3) * BK + kk].cast[F32]()
            ) * bv
        barrier()
    var cb = z * sc
    c.unsafe_store(cb + (row0 + 0) * n + col0, acc0)
    c.unsafe_store(cb + (row0 + 1) * n + col0, acc1)
    c.unsafe_store(cb + (row0 + 2) * n + col0, acc2)
    c.unsafe_store(cb + (row0 + 3) * n + col0, acc3)


def softmax_rows_z(
    scores: Pointer[Scalar[F32], MutAnyOrigin],
    sums: Pointer[Scalar[F32], MutAnyOrigin],
    rows_dev: Int32, n_valid_dev: Int32, n_pad_dev: Int32,
):
    """Per row of scores[unsafe_offset=z] [rows, n_pad]: m = max over the first n_valid
    columns, p = exp(x - m) in place (0 in the padded tail), row sum ->
    sums[z * rows + r]. The 1/sum is fused into the CPU readback. One
    thread per row."""
    var rows = Int(rows_dev)
    var n_valid = Int(n_valid_dev)
    var n_pad = Int(n_pad_dev)

    var z = Int(block_idx.z)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= rows:
        return
    var base = z * rows * n_pad + r * n_pad
    var m = scores[unsafe_offset=base]
    for j in range(1, n_valid):
        var x = scores[unsafe_offset=base + j]
        if x > m:
            m = x
    var s: Float32 = 0
    for j in range(n_valid):
        var p = exp(scores[unsafe_offset=base + j] - m)
        scores[unsafe_offset=base + j] = p
        s += p
    for j in range(n_valid, n_pad):
        scores[unsafe_offset=base + j] = 0
    sums[unsafe_offset=z * rows + r] = s


def gpu_sdpa_wants(l: Int, lkv: Int, head_dim: Int, heads: Int) -> Bool:
    """Shape gate for the GPU path: the head dim must be a full GEMM tile
    column (n % 64 for av) and ONE HEAD's scores must fit the scratch cap
    (WP17: the composition runs in head groups, so the per-call limit is
    per head — heads only affect how many groups get enqueued); small
    shapes lose to the CPU flash path on fixed overhead. Both q and kv
    lengths are padded to 64 internally, so any lengths pass."""
    if l < GPU_SDPA_MIN_Q or lkv < GPU_SDPA_MIN_KV:
        return False
    if head_dim % BN != 0 or head_dim % BK != 0:
        return False
    var m_pad = ((l + BM - 1) // BM) * BM
    var lkv_pad = ((lkv + BN - 1) // BN) * BN
    return m_pad * lkv_pad <= GPU_SDPA_MAX_SCORES


def _sdpa_sc_need(h: Int, mp: Int, lp: Int) -> Int:
    """Scores-scratch floats for the head-grouped composition: as many
    heads per group as fit under GPU_SDPA_MAX_SCORES (>= 1 by the gate)."""
    var hg = GPU_SDPA_MAX_SCORES // (mp * lp)
    if hg > h:
        hg = h
    if hg < 1:
        hg = 1
    return hg * mp * lp


def _enqueue_sdpa_groups(
    g: GpuContext,
    qh: Pointer[Scalar[F32], _],
    kt: Pointer[Scalar[F32], _],
    vh: Pointer[Scalar[F32], _],
    sc: Pointer[Scalar[F32], _],
    su: Pointer[Scalar[F32], _],
    ob: Pointer[Scalar[F32], _],
    h: Int, mp: Int, lp: Int, lkv: Int, d: Int,
) raises:
    """WP17: the qk -> masked-softmax -> av composition enqueued in HEAD
    GROUPS so the scores scratch never exceeds GPU_SDPA_MAX_SCORES floats.
    Motivation is the SIXTH b2-Metal law (probed 2026-07-11): kernel
    writes past a 4 GiB byte offset within one buffer binding are
    SILENTLY LOST, so a full-H scores buffer for the 1024-cascade slat
    (T ~ 12k -> 9 GB) can never work as one binding. The queue is
    in-order, so reusing the scores buffer across groups is safe; per-z
    strides make a group just base-pointer offsets + grid z = group
    size. One group (hg == h) reproduces the old single-enqueue path
    exactly."""
    var hg = GPU_SDPA_MAX_SCORES // (mp * lp)
    if hg > h:
        hg = h
    if hg < 1:
        hg = 1

    var z0 = 0
    while z0 < h:
        var zn = min(hg, h - z0)
        var qh_a = rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](qh.unsafe_offset(z0 * mp * d))
        var kt_b = rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](kt.unsafe_offset(z0 * d * lp))
        var sc_c = rebind[Pointer[Scalar[F32], MutAnyOrigin]](sc)
        if g.f16:
            g.ctx.enqueue_function[gemm_z_f16sh](
                qh_a, kt_b, sc_c,
                Int32(lp), Int32(d), Int32(mp * d), Int32(d * lp), Int32(mp * lp),
                grid_dim=(lp // BN, mp // BM, zn), block_dim=(16, 16),
            )
        else:
            g.ctx.enqueue_function[gemm_z](
                qh_a, kt_b, sc_c,
                Int32(lp), Int32(d), Int32(mp * d), Int32(d * lp), Int32(mp * lp),
                grid_dim=(lp // BN, mp // BM, zn), block_dim=(16, 16),
            )
        var sc_s = rebind[Pointer[Scalar[F32], MutAnyOrigin]](sc)
        var su_s = rebind[Pointer[Scalar[F32], MutAnyOrigin]](su.unsafe_offset(z0 * mp))
        g.ctx.enqueue_function[softmax_rows_z](
            sc_s, su_s,
            Int32(mp), Int32(lkv), Int32(lp),
            grid_dim=((mp + 255) // 256, 1, zn), block_dim=(256, 1, 1),
        )
        var sc_a = rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](sc)
        var vh_b = rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](vh.unsafe_offset(z0 * lp * d))
        var ob_c = rebind[Pointer[Scalar[F32], MutAnyOrigin]](ob.unsafe_offset(z0 * mp * d))
        if g.f16:
            g.ctx.enqueue_function[gemm_z_f16sh](
                sc_a, vh_b, ob_c,
                Int32(d), Int32(lp), Int32(mp * lp), Int32(lp * d), Int32(mp * d),
                grid_dim=(d // BN, mp // BM, zn), block_dim=(16, 16),
            )
        else:
            g.ctx.enqueue_function[gemm_z](
                sc_a, vh_b, ob_c,
                Int32(d), Int32(lp), Int32(mp * lp), Int32(lp * d), Int32(mp * d),
                grid_dim=(d // BN, mp // BM, zn), block_dim=(16, 16),
            )
        z0 += zn


def gpu_dense_sdpa(
    g: GpuContext,
    q: Tensor[F32],
    k: Tensor[F32],
    v: Tensor[F32],
) raises -> Tensor[F32]:
    """q [1, L, H, D], k/v [1, Lkv, H, D] -> [1, L, H, D]. Same contract
    as dense_sdpa_q_k_v for batch 1 (the DiT runner is always N=1);
    callers gate on gpu_sdpa_wants first."""
    if q.shape[0] != 1:
        raise Error("gpu_dense_sdpa: batch must be 1")
    var l = q.shape[1]
    var h = q.shape[2]
    var d = q.shape[3]
    var lkv = k.shape[1]
    if k.shape[2] != h or v.shape[2] != h or k.shape[3] != d or v.shape[3] != d:
        raise Error("gpu_dense_sdpa: head/channel mismatch")
    var out_shape: List[Int] = [1, l, h, d]
    var out = Tensor[F32](out_shape)
    _sdpa_core(g, q, k, v, l, lkv, h, d, out)
    return out^


def gpu_varlen_sdpa_single(
    g: GpuContext,
    q: Tensor[F32],
    k: Tensor[F32],
    v: Tensor[F32],
) raises -> Tensor[F32]:
    """Single-segment varlen: q [T, H, D], k/v [Tkv, H, D] -> [T, H, D]
    (varlen_sdpa's layout with offsets [0, T]/[0, Tkv] — the B=1 sparse
    DiT case). Arbitrary T on both sides; callers gate on
    gpu_sdpa_wants first."""
    var l = q.shape[0]
    var h = q.shape[1]
    var d = q.shape[2]
    var lkv = k.shape[0]
    if k.shape[1] != h or v.shape[1] != h or k.shape[2] != d or v.shape[2] != d:
        raise Error("gpu_varlen_sdpa_single: head/channel mismatch")
    var out_shape: List[Int] = [l, h, d]
    var out = Tensor[F32](out_shape)
    _sdpa_core(g, q, k, v, l, lkv, h, d, out)
    return out^


def _sdpa_core(
    g: GpuContext,
    q: Tensor[F32],
    k: Tensor[F32],
    v: Tensor[F32],
    l: Int, lkv: Int, h: Int, d: Int,
    mut out: Tensor[F32],
) raises:
    """The shared composition on token-major [.., H, D] data. q rows pad
    to m_pad with zeros (their outputs are dropped in the readback), kv
    pads to lp with zero k-columns/v-rows plus the softmax mask."""
    var lp = ((lkv + BN - 1) // BN) * BN
    var mp = ((l + BM - 1) // BM) * BM
    var scale = Float32(1.0 / Float64(d) ** 0.5)

    var s = g.attn
    if s[].qh_cap < h * mp * d:
        s[].qh = g.ctx.enqueue_create_buffer[F32](h * mp * d)
        s[].qh_cap = h * mp * d
    if s[].kt_cap < h * d * lp:
        s[].kt = g.ctx.enqueue_create_buffer[F32](h * d * lp)
        s[].kt_cap = h * d * lp
    if s[].vh_cap < h * lp * d:
        s[].vh = g.ctx.enqueue_create_buffer[F32](h * lp * d)
        s[].vh_cap = h * lp * d
    var sc_need = _sdpa_sc_need(h, mp, lp)
    if s[].sc_cap < sc_need:
        s[].sc = g.ctx.enqueue_create_buffer[F32](sc_need)
        s[].sc_cap = sc_need
    if s[].ob_cap < h * mp * d:
        s[].ob = g.ctx.enqueue_create_buffer[F32](h * mp * d)
        s[].ob_cap = h * mp * d
    if s[].su_cap < h * mp:
        s[].su = g.ctx.enqueue_create_buffer[F32](h * mp)
        s[].su_cap = h * mp

    # pack per head (parallel over heads): q scaled with zero pad rows
    # (their scores become 0 -> softmax stays finite -> outputs dropped),
    # k transposed with zero-padded columns, v with zero-padded rows
    var qp = q.data.unsafe_ptr()
    var kp = k.data.unsafe_ptr()
    var vp = v.data.unsafe_ptr()
    with s[].qh.value().map_to_host() as hq:
        with s[].kt.value().map_to_host() as hk:
            with s[].vh.value().map_to_host() as hv:
                var qhp = hq.unsafe_ptr()
                var ktp = hk.unsafe_ptr()
                var vhp = hv.unsafe_ptr()

                @parameter
                def pack_head(head: Int):
                    var qb = head * mp * d
                    for t in range(l):
                        var src = (t * h + head) * d
                        var dst = qb + t * d
                        for e in range(d):
                            qhp[unsafe_offset=(dst + e)] = qp[unsafe_offset=src + e] * scale
                    for i in range(l * d, mp * d):
                        qhp[unsafe_offset=qb + i] = 0
                    var kb = head * d * lp
                    for t in range(lkv):
                        var src = (t * h + head) * d
                        for e in range(d):
                            ktp[unsafe_offset=kb + e * lp + t] = kp[unsafe_offset=src + e]
                    for t in range(lkv, lp):
                        for e in range(d):
                            ktp[unsafe_offset=kb + e * lp + t] = 0
                    var vb = head * lp * d
                    for t in range(lkv):
                        var src = (t * h + head) * d
                        var dst = vb + t * d
                        for e in range(d):
                            vhp[unsafe_offset=(dst + e)] = vp[unsafe_offset=src + e]
                    for i in range(lkv * d, lp * d):
                        vhp[unsafe_offset=vb + i] = 0

                parallelize[pack_head](h)

    # the composition, in head groups (WP17 — see _enqueue_sdpa_groups)
    _enqueue_sdpa_groups(
        g,
        s[].qh.value().unsafe_ptr(), s[].kt.value().unsafe_ptr(),
        s[].vh.value().unsafe_ptr(), s[].sc.value().unsafe_ptr(),
        s[].su.value().unsafe_ptr(), s[].ob.value().unsafe_ptr(),
        h, mp, lp, lkv, d,
    )
    g.barrier()

    # softmax denominators to cached heap first (scattered 4-byte reads
    # from WC memory would be one transaction each)
    var sums = List[Float32](length=h * mp, fill=0)
    with s[].su.value().map_to_host() as hs:
        unsafe_memcpy(dest=sums.unsafe_ptr(), src=hs.unsafe_ptr(), count=h * mp)

    # readback: re-interleave [H, mp, D] -> [T, H, D] with the denominator
    # fused in, skipping the pad rows. Work items are (head, t-range) so
    # the WC reads stream sequentially within each head; the strided heap
    # writes are cached.
    var op = out.data.unsafe_ptr()
    var sup = sums.unsafe_ptr()
    with s[].ob.value().map_to_host() as ho:
        var obp = ho.unsafe_ptr()
        comptime W = 8
        comptime NT = 4
        var chunk = (l + NT - 1) // NT

        @parameter
        def copy_out(item: Int):
            var head = item // NT
            var t0 = (item % NT) * chunk
            var t1 = min(t0 + chunk, l)
            for t in range(t0, t1):
                var inv = Float32(1.0) / sup[unsafe_offset=head * mp + t]
                var src = (head * mp + t) * d
                var dst = (t * h + head) * d
                var e = 0
                while e + W <= d:
                    op.unsafe_store((dst + e), obp.unsafe_load[width=W]((src + e)) * inv)
                    e += W
                while e < d:
                    op[unsafe_offset=(dst + e)] = obp[unsafe_offset=src + e] * inv
                    e += 1

        parallelize[copy_out](h * NT)


# -- WP11 step 7: device-resident self-attention chain -------------------------


def bias_rms_rope_qkv(
    qkv: Pointer[Scalar[F32], MutAnyOrigin],
    consts: Pointer[Scalar[F32], ImmutAnyOrigin],
    phases: Pointer[Scalar[F32], ImmutAnyOrigin],
    rows_dev: Int32, h_dev: Int32, d_dev: Int32, use_rms_dev: Int32, use_rope_dev: Int32,
):
    """In place on the qkv GEMM output [rows, 3, H, D] (valid rows only —
    pad rows are zeroed by the pack kernels): add the qkv bias, then
    rms-normalize q/k with their gammas, then rotate q/k by the rope
    phases [rows, D/2, 2]. consts = [3HD bias][HD gamma_q][HD gamma_k]
    (ONE buffer keeps the kernel at 3 pointers — marshalling law). One
    thread per (row, head); per-element formulas match the CPU
    MultiHeadRMSNorm / _rotate exactly, only accumulation order differs."""
    var rows = Int(rows_dev)
    var h = Int(h_dev)
    var d = Int(d_dev)
    var use_rms = Int(use_rms_dev)
    var use_rope = Int(use_rope_dev)

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= rows * h:
        return
    var row = i // h
    var head = i % h
    var hd = h * d
    var stride = 3 * hd
    for p in range(3):
        var off = row * stride + p * hd + head * d
        var boff = p * hd + head * d
        for e in range(d):
            qkv[unsafe_offset=off + e] = qkv[unsafe_offset=off + e] + consts[unsafe_offset=boff + e]
    if use_rms == 1:
        var sc = sqrt(Float32(d))
        for p in range(2):
            var off = row * stride + p * hd + head * d
            var goff = 3 * hd + p * hd + head * d
            var acc: Float32 = 0
            for e in range(d):
                var v = qkv[unsafe_offset=off + e]
                acc += v * v
            var norm = sqrt(acc)
            if norm < 1e-12:
                norm = 1e-12
            for e in range(d):
                qkv[unsafe_offset=off + e] = qkv[unsafe_offset=off + e] / norm * consts[unsafe_offset=goff + e] * sc
    if use_rope == 1:
        var half = d // 2
        var pbase = row * half * 2
        for p in range(2):
            var off = row * stride + p * hd + head * d
            for pr in range(half):
                var c = phases[unsafe_offset=pbase + 2 * pr]
                var s = phases[unsafe_offset=pbase + 2 * pr + 1]
                var re = qkv[unsafe_offset=off + 2 * pr]
                var im = qkv[unsafe_offset=off + 2 * pr + 1]
                qkv[unsafe_offset=off + 2 * pr] = re * c - im * s
                qkv[unsafe_offset=off + 2 * pr + 1] = re * s + im * c


def pack_q_z(
    qkv: Pointer[Scalar[F32], ImmutAnyOrigin],
    qh: Pointer[Scalar[F32], MutAnyOrigin],
    l_dev: Int32, h_dev: Int32, d_dev: Int32, mp_dev: Int32, stride_dev: Int32,
):
    """q rows at `stride` elements apart (3*H*D for the fused qkv buffer,
    H*D for the cross q buffer) -> qh [H, mp, D] with the sdpa scale
    baked in (recomputed from d — no float scalars); pad rows zeroed so
    their softmax stays finite. grid z = head, thread = row."""
    var l = Int(l_dev)
    var h = Int(h_dev)
    var d = Int(d_dev)
    var mp = Int(mp_dev)
    var stride = Int(stride_dev)

    var z = Int(block_idx.z)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= mp:
        return
    var dst = (z * mp + t) * d
    if t < l:
        var scale = Float32(1.0) / sqrt(Float32(d))
        var src = t * stride + z * d
        for e in range(d):
            qh[unsafe_offset=(dst + e)] = qkv[unsafe_offset=src + e] * scale
    else:
        for e in range(d):
            qh[unsafe_offset=(dst + e)] = 0


def bias_rms_q(
    q: Pointer[Scalar[F32], MutAnyOrigin],
    consts: Pointer[Scalar[F32], ImmutAnyOrigin],
    rows_dev: Int32, h_dev: Int32, d_dev: Int32, use_rms_dev: Int32,
):
    """Cross-chain epilogue: in place on the q GEMM output [rows, H, D]
    (valid rows only), add the q bias and rms-normalize with gamma_q.
    consts = [HD bias][HD gamma_q]. One thread per (row, head); formulas
    match bias_rms_rope_qkv's q part."""
    var rows = Int(rows_dev)
    var h = Int(h_dev)
    var d = Int(d_dev)
    var use_rms = Int(use_rms_dev)

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= rows * h:
        return
    var row = i // h
    var head = i % h
    var hd = h * d
    var off = row * hd + head * d
    var boff = head * d
    for e in range(d):
        q[unsafe_offset=off + e] = q[unsafe_offset=off + e] + consts[unsafe_offset=boff + e]
    if use_rms == 1:
        var sc = sqrt(Float32(d))
        var goff = hd + head * d
        var acc: Float32 = 0
        for e in range(d):
            var v = q[unsafe_offset=off + e]
            acc += v * v
        var norm = sqrt(acc)
        if norm < 1e-12:
            norm = 1e-12
        for e in range(d):
            q[unsafe_offset=off + e] = q[unsafe_offset=off + e] / norm * consts[unsafe_offset=goff + e] * sc


def pack_kv_z(
    qkv: Pointer[Scalar[F32], ImmutAnyOrigin],
    kt: Pointer[Scalar[F32], MutAnyOrigin],
    vh: Pointer[Scalar[F32], MutAnyOrigin],
    l_dev: Int32, h_dev: Int32, d_dev: Int32, lp_dev: Int32,
):
    """qkv parts 1/2 -> kt [H, D, lp] (transposed) + vh [H, lp, D]; pad
    columns/rows zeroed (v pads MUST be zero: the av GEMM multiplies them
    by the zeroed probs and 0*NaN would poison the output)."""
    var l = Int(l_dev)
    var h = Int(h_dev)
    var d = Int(d_dev)
    var lp = Int(lp_dev)

    var z = Int(block_idx.z)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= lp:
        return
    var hd = h * d
    var kb = z * d * lp
    var vdst = (z * lp + t) * d
    if t < l:
        var ksrc = t * 3 * hd + hd + z * d
        var vsrc = t * 3 * hd + 2 * hd + z * d
        for e in range(d):
            kt[unsafe_offset=kb + e * lp + t] = qkv[unsafe_offset=ksrc + e]
            vh[unsafe_offset=vdst + e] = qkv[unsafe_offset=vsrc + e]
    else:
        for e in range(d):
            kt[unsafe_offset=kb + e * lp + t] = 0
            vh[unsafe_offset=vdst + e] = 0


def unpack_o_z(
    ob: Pointer[Scalar[F32], ImmutAnyOrigin],
    su: Pointer[Scalar[F32], ImmutAnyOrigin],
    dst: Pointer[Scalar[F32], MutAnyOrigin],
    mp_dev: Int32, h_dev: Int32, d_dev: Int32,
):
    """ob [H, mp, D] -> dst [mp, H*D] with the softmax 1/sum fused. ALL
    mp rows are written (pad rows are finite: zero q rows give a uniform
    softmax) so the out-GEMM reads no stale scratch."""
    var mp = Int(mp_dev)
    var h = Int(h_dev)
    var d = Int(d_dev)

    var z = Int(block_idx.z)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= mp:
        return
    var inv = Float32(1.0) / su[unsafe_offset=z * mp + t]
    var src = (z * mp + t) * d
    var doff = t * h * d + z * d
    for e in range(d):
        dst[unsafe_offset=doff + e] = ob[unsafe_offset=src + e] * inv


struct GpuAttnChain(Copyable, Movable):
    """Per-MHA device constants for the chained attention paths. Self
    (try_build): the qkv bias and both rms gammas in ONE buffer
    ([3HD][HD][HD]). Cross (try_build_cross, WP11 step 8): the q bias and
    gamma_q ([HD][HD]) — k-rms runs on the CPU kv path. Uploaded once at
    model load by dense_mha_from/sparse_mha_from."""

    var g: GpuContext
    var consts: DeviceBuffer[F32]
    var h: Int
    var d: Int
    var use_rms: Bool
    var is_cross: Bool

    def __init__(
        out self,
        var g: GpuContext,
        var consts: DeviceBuffer[F32],
        h: Int,
        d: Int,
        use_rms: Bool,
        is_cross: Bool = False,
    ):
        self.g = g^
        self.consts = consts^
        self.h = h
        self.d = d
        self.use_rms = use_rms
        self.is_cross = is_cross

    @staticmethod
    def try_build(
        gpu: Optional[GpuContext],
        qkv: Optional[GpuLinear],
        out_lin: Optional[GpuLinear],
        gamma_q: Tensor[F32],
        gamma_k: Tensor[F32],
        use_rms: Bool,
        h: Int,
        d: Int,
    ) raises -> Optional[GpuAttnChain]:
        """None unless both linears have device weights and the shapes fit
        the composition (d is a full GEMM tile: %64). Cross-attention MHAs
        have a dummy to_qkv without device weights -> None automatically."""
        if not gpu or not qkv or not out_lin:
            return None
        if d % BN != 0:
            return None
        var hd = h * d
        if qkv.value().ci != hd or qkv.value().co != 3 * hd or out_lin.value().ci != hd:
            return None
        var g = gpu.value().copy()
        var consts = g.ctx.enqueue_create_buffer[F32](5 * hd)
        with consts.map_to_host() as hm:
            var p = hm.unsafe_ptr()
            if qkv.value().has_bias:
                unsafe_memcpy(dest=p, src=qkv.value().bias_host.unsafe_ptr(), count=3 * hd)
            else:
                for i in range(3 * hd):
                    p[unsafe_offset=i] = 0
            if use_rms:
                unsafe_memcpy(dest=p.unsafe_offset(3 * hd), src=gamma_q.data.unsafe_ptr(), count=hd)
                unsafe_memcpy(dest=p.unsafe_offset(4 * hd), src=gamma_k.data.unsafe_ptr(), count=hd)
            else:
                for i in range(2 * hd):
                    p[unsafe_offset=3 * hd + i] = 1
        return GpuAttnChain(g^, consts^, h, d, use_rms)

    @staticmethod
    def try_build_cross(
        gpu: Optional[GpuContext],
        q_lin: Optional[GpuLinear],
        out_lin: Optional[GpuLinear],
        gamma_q: Tensor[F32],
        use_rms: Bool,
        h: Int,
        d: Int,
    ) raises -> Optional[GpuAttnChain]:
        """Cross variant (WP11 step 8): only the q side chains on the GPU
        (kv is computed and rms-normalized on the CPU and host-packed).
        consts = [HD q-bias][HD gamma_q]."""
        if not gpu or not q_lin or not out_lin:
            return None
        if d % BN != 0:
            return None
        var hd = h * d
        if q_lin.value().ci != hd or q_lin.value().co != hd or out_lin.value().ci != hd:
            return None
        var g = gpu.value().copy()
        var consts = g.ctx.enqueue_create_buffer[F32](2 * hd)
        with consts.map_to_host() as hm:
            var p = hm.unsafe_ptr()
            if q_lin.value().has_bias:
                unsafe_memcpy(dest=p, src=q_lin.value().bias_host.unsafe_ptr(), count=hd)
            else:
                for i in range(hd):
                    p[unsafe_offset=i] = 0
            if use_rms:
                unsafe_memcpy(dest=p.unsafe_offset(hd), src=gamma_q.data.unsafe_ptr(), count=hd)
            else:
                for i in range(hd):
                    p[unsafe_offset=hd + i] = 1
        return GpuAttnChain(g^, consts^, h, d, use_rms, is_cross=True)

    def wants(self, rows: Int, qkv: GpuLinear) -> Bool:
        """Per-call gate: the sdpa shape gate plus the qkv GEMM's own
        flops threshold (the out-GEMM rides along for free — its solo
        break-even does not apply inside the chain)."""
        return gpu_sdpa_wants(rows, rows, self.d, self.h) and qkv.wants(rows)

    def wants_cross(self, l: Int, lkv: Int) -> Bool:
        """Cross gate: the sdpa shape gate alone — the q/out GEMMs ride on
        the upload the sdpa needs anyway, so their solo thresholds do not
        apply (verified on both the ss and slat cross geometries)."""
        return gpu_sdpa_wants(l, lkv, self.d, self.h)


def _upload_phases(g: GpuContext, phases: Tensor[F32], rows: Int, d: Int) raises -> DeviceBuffer[F32]:
    """Upload rope phases [rows, D/2, 2] to the phases scratch. Mapping the
    host-written buffer commits pending queue work — callers upload BEFORE
    enqueueing (block orchestration relies on this ordering)."""
    if phases.shape[0] != rows or phases.shape[1] != d // 2:
        raise Error("gpu attention chain: phases shape mismatch")
    var t = g.attn
    var need = rows * (d // 2) * 2
    if t[].ph_cap < need:
        t[].ph = g.ctx.enqueue_create_buffer[F32](need)
        t[].ph_cap = need
    with t[].ph.value().map_to_host() as hp:
        unsafe_memcpy(dest=hp.unsafe_ptr(), src=phases.data.unsafe_ptr(), count=need)
    return t[].ph.value()


def gpu_attn_self_chain(
    chain: GpuAttnChain,
    x: Tensor[F32],
    qkv_lin: GpuLinear,
    out_lin: GpuLinear,
) raises -> Tensor[F32]:
    """Chained self-attention without rope; x [.., C] with C = H*D."""
    return _attn_chain_core(chain, x, qkv_lin, out_lin, False, chain.consts)


def gpu_attn_self_chain(
    chain: GpuAttnChain,
    x: Tensor[F32],
    qkv_lin: GpuLinear,
    out_lin: GpuLinear,
    phases: Tensor[F32],
) raises -> Tensor[F32]:
    """Chained self-attention with rope phases [rows, D/2, 2] (both the
    dense flow model's precomputed grid and the sparse embedder's
    per-coords phases use this layout)."""
    var rows = x.numel() // x.shape[len(x.shape) - 1]
    var g = chain.g.copy()
    var ph = _upload_phases(g, phases, rows, chain.d)
    return _attn_chain_core(chain, x, qkv_lin, out_lin, True, ph)


def _attn_chain_core(
    chain: GpuAttnChain,
    x: Tensor[F32],
    qkv_lin: GpuLinear,
    out_lin: GpuLinear,
    use_rope: Bool,
    ph_buf: DeviceBuffer[F32],
) raises -> Tensor[F32]:
    """Host wrapper: one upload (x), the enqueued chain, one barrier, one
    readback (out + out-bias). Callers gate on chain.wants first."""
    var hd = chain.h * chain.d
    var ci = x.shape[len(x.shape) - 1]
    if ci != hd or qkv_lin.ci != hd or out_lin.ci != hd:
        raise Error("gpu_attn_self_chain: channel mismatch")
    var rows = x.numel() // ci
    var mp = ((rows + BM - 1) // BM) * BM
    var out_shape = x.shape.copy()
    out_shape[len(out_shape) - 1] = out_lin.co
    var out = Tensor[F32](out_shape)

    var g = chain.g.copy()
    var s = g.scratch
    var a_need = mp * max(ci, out_lin.co)
    if s[].a_cap < a_need:
        s[].a = g.ctx.enqueue_create_buffer[F32](a_need)
        s[].a_cap = a_need

    # upload x (chunked parallel memcpy; pad rows stay stale — every
    # consumer either skips or zeroes them)
    comptime NCHUNK = 16
    with s[].a.value().map_to_host() as hm:
        var ap = hm.unsafe_ptr()
        var xp = x.data.unsafe_ptr()
        var total = rows * ci
        var chunk = (total + NCHUNK - 1) // NCHUNK

        @parameter
        def copy_in(i: Int):
            var lo = i * chunk
            var n = min(chunk, total - lo)
            if n > 0:
                unsafe_memcpy(dest=ap.unsafe_offset(lo), src=xp.unsafe_offset(lo), count=n)

        parallelize[copy_in](NCHUNK)

    _attn_chain_enqueue(
        chain, g, s[].a.value(), rows, qkv_lin, out_lin, use_rope, ph_buf,
        s[].a.value(),
    )
    g.barrier()

    # readback with the out-bias fused (GpuLinear.forward's pattern)
    var op = out.data.unsafe_ptr()
    var co = out_lin.co
    var has_bias = out_lin.has_bias
    var bp = out_lin.bias_host.unsafe_ptr()
    with s[].a.value().map_to_host() as hm:
        var hp = hm.unsafe_ptr()
        var rchunk = (rows + NCHUNK - 1) // NCHUNK

        @parameter
        def copy_out(i: Int):
            var r0 = i * rchunk
            var r1 = min(r0 + rchunk, rows)
            if r1 <= r0:
                return
            if has_bias:
                comptime W = 8
                var co_main = (co // W) * W
                for r in range(r0, r1):
                    var base = r * co
                    var j = 0
                    while j < co_main:
                        op.unsafe_store(base + j, hp.unsafe_load[width=W](base + j) + bp.unsafe_load[width=W](j))
                        j += W
                    while j < co:
                        op[unsafe_offset=base + j] = hp[unsafe_offset=base + j] + bp[unsafe_offset=j]
                        j += 1
            else:
                unsafe_memcpy(dest=op.unsafe_offset(r0 * co), src=hp.unsafe_offset(r0 * co), count=(r1 - r0) * co)

        parallelize[copy_out](NCHUNK)
    return out^


def _attn_chain_enqueue(
    chain: GpuAttnChain,
    g: GpuContext,
    in_buf: DeviceBuffer[F32],
    rows: Int,
    qkv_lin: GpuLinear,
    out_lin: GpuLinear,
    use_rope: Bool,
    ph_buf: DeviceBuffer[F32],
    out_buf: DeviceBuffer[F32],
) raises:
    """Enqueue-only self-attention chain from a device-resident input
    [mp, C] to a device-resident output [mp, C] (in_buf == out_buf is
    fine — in_buf is free after the qkv GEMM, the queue is in-order).
    Input pad rows may be garbage (the packs zero them); output pad rows
    are garbage. The out-linear bias is NOT added: the host wrapper fuses
    it in the readback, the block path folds it into gate_add. No
    transfers, no barrier."""
    var h = chain.h
    var d = chain.d
    var hd = h * d
    var mp = ((rows + BM - 1) // BM) * BM
    var s = g.scratch
    if s[].c_cap < mp * qkv_lin.co:
        s[].c = g.ctx.enqueue_create_buffer[F32](mp * qkv_lin.co)
        s[].c_cap = mp * qkv_lin.co
    var t = g.attn
    if t[].qh_cap < h * mp * d:
        t[].qh = g.ctx.enqueue_create_buffer[F32](h * mp * d)
        t[].qh_cap = h * mp * d
    if t[].kt_cap < h * d * mp:
        t[].kt = g.ctx.enqueue_create_buffer[F32](h * d * mp)
        t[].kt_cap = h * d * mp
    if t[].vh_cap < h * mp * d:
        t[].vh = g.ctx.enqueue_create_buffer[F32](h * mp * d)
        t[].vh_cap = h * mp * d
    var sc_need = _sdpa_sc_need(h, mp, mp)
    if t[].sc_cap < sc_need:
        t[].sc = g.ctx.enqueue_create_buffer[F32](sc_need)
        t[].sc_cap = sc_need
    if t[].ob_cap < h * mp * d:
        t[].ob = g.ctx.enqueue_create_buffer[F32](h * mp * d)
        t[].ob_cap = h * mp * d
    if t[].su_cap < h * mp:
        t[].su = g.ctx.enqueue_create_buffer[F32](h * mp)
        t[].su_cap = h * mp

    # qkv GEMM: c [mp, 3C] = in [mp, C] @ Wqkv^T (dispatches on the weight
    # storage format — WP11 step 14)
    qkv_lin.enqueue_gemm(
        in_buf.unsafe_ptr(), s[].c.value().unsafe_ptr(), mp
    )
    # bias + rms + rope on the valid rows, in place
    g.ctx.enqueue_function[bias_rms_rope_qkv](
        rebind[Pointer[Scalar[F32], MutAnyOrigin]](s[].c.value().unsafe_ptr()),
        rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](chain.consts.unsafe_ptr()),
        rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](ph_buf.unsafe_ptr()),
        Int32(rows),
        Int32(h),
        Int32(d),
        Int32(1 if chain.use_rms else 0),
        Int32(1 if use_rope else 0),
        grid_dim=((rows * h + 255) // 256,),
        block_dim=(256,),
    )
    # head-major pack (q scaled, kv transposed/zero-padded)
    g.ctx.enqueue_function[pack_q_z](
        rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](s[].c.value().unsafe_ptr()),
        rebind[Pointer[Scalar[F32], MutAnyOrigin]](t[].qh.value().unsafe_ptr()),
        Int32(rows), Int32(h), Int32(d), Int32(mp), Int32(3 * hd),
        grid_dim=((mp + 255) // 256, 1, h), block_dim=(256, 1, 1),
    )
    g.ctx.enqueue_function[pack_kv_z](
        rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](s[].c.value().unsafe_ptr()),
        rebind[Pointer[Scalar[F32], MutAnyOrigin]](t[].kt.value().unsafe_ptr()),
        rebind[Pointer[Scalar[F32], MutAnyOrigin]](t[].vh.value().unsafe_ptr()),
        Int32(rows), Int32(h), Int32(d), Int32(mp),
        grid_dim=((mp + 255) // 256, 1, h), block_dim=(256, 1, 1),
    )
    # the SDPA composition (identical to _sdpa_core's), in head groups
    _enqueue_sdpa_groups(
        g,
        t[].qh.value().unsafe_ptr(), t[].kt.value().unsafe_ptr(),
        t[].vh.value().unsafe_ptr(), t[].sc.value().unsafe_ptr(),
        t[].su.value().unsafe_ptr(), t[].ob.value().unsafe_ptr(),
        h, mp, mp, rows, d,
    )
    # re-interleave into the C scratch (free: the packs consumed it and
    # the queue is in-order), then the out-GEMM into out_buf
    g.ctx.enqueue_function[unpack_o_z](
        rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](t[].ob.value().unsafe_ptr()),
        rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](t[].su.value().unsafe_ptr()),
        rebind[Pointer[Scalar[F32], MutAnyOrigin]](s[].c.value().unsafe_ptr()),
        Int32(mp), Int32(h), Int32(d),
        grid_dim=((mp + 255) // 256, 1, h), block_dim=(256, 1, 1),
    )
    out_lin.enqueue_gemm(
        s[].c.value().unsafe_ptr(), out_buf.unsafe_ptr(), mp
    )


def _cross_pack_kv(
    g: GpuContext, k: Tensor[F32], v: Tensor[F32], h: Int, d: Int, lp: Int
) raises:
    """Host-pack the CPU-computed kv [Lkv, H, D] into the DEDICATED cross
    buffers ckt [H, D, lp] / cvh [H, lp, D] with zero pads (v pads MUST
    be zero — 0*NaN would poison the av GEMM). Mapping the host-written
    buffers commits pending queue work, so callers pack BEFORE enqueueing;
    the buffers are separate from kt/vh because the self chain's
    device-side pack owns those in the fused block queue."""
    var lkv = k.shape[0]
    var t = g.attn
    if t[].ckt_cap < h * d * lp:
        t[].ckt = g.ctx.enqueue_create_buffer[F32](h * d * lp)
        t[].ckt_cap = h * d * lp
    if t[].cvh_cap < h * lp * d:
        t[].cvh = g.ctx.enqueue_create_buffer[F32](h * lp * d)
        t[].cvh_cap = h * lp * d
    var kp = k.data.unsafe_ptr()
    var vp = v.data.unsafe_ptr()
    with t[].ckt.value().map_to_host() as hk:
        with t[].cvh.value().map_to_host() as hv:
            var ktp = hk.unsafe_ptr()
            var vhp = hv.unsafe_ptr()

            @parameter
            def pack_head(head: Int):
                var kb = head * d * lp
                for tt in range(lkv):
                    var src = (tt * h + head) * d
                    for e in range(d):
                        ktp[unsafe_offset=kb + e * lp + tt] = kp[unsafe_offset=src + e]
                for tt in range(lkv, lp):
                    for e in range(d):
                        ktp[unsafe_offset=kb + e * lp + tt] = 0
                var vb = head * lp * d
                for tt in range(lkv):
                    var src = (tt * h + head) * d
                    var dst = vb + tt * d
                    for e in range(d):
                        vhp[unsafe_offset=(dst + e)] = vp[unsafe_offset=src + e]
                for i in range(lkv * d, lp * d):
                    vhp[unsafe_offset=vb + i] = 0

            parallelize[pack_head](h)


def gpu_attn_cross_chain(
    chain: GpuAttnChain,
    x: Tensor[F32],
    k: Tensor[F32],
    v: Tensor[F32],
    q_lin: GpuLinear,
    out_lin: GpuLinear,
) raises -> Tensor[F32]:
    """WP11 step 8: chained cross-attention — the q side runs
    q-GEMM -> bias+q-rms -> pack -> sdpa -> unpack -> out-GEMM
    device-resident; k/v [Lkv, H, D] are computed (and k-rms-normalized)
    on the CPU by the caller and host-packed (the kv linear at ~1k
    context rows is below the GPU GEMM's break-even). Callers gate on
    chain.wants_cross first."""
    var h = chain.h
    var d = chain.d
    var hd = h * d
    var ci = x.shape[len(x.shape) - 1]
    if ci != hd or q_lin.ci != hd or out_lin.ci != hd:
        raise Error("gpu_attn_cross_chain: channel mismatch")
    if k.shape[1] != h or k.shape[2] != d or v.shape[1] != h or v.shape[2] != d:
        raise Error("gpu_attn_cross_chain: kv head/channel mismatch")
    var lkv = k.shape[0]
    if v.shape[0] != lkv:
        raise Error("gpu_attn_cross_chain: k/v length mismatch")
    var rows = x.numel() // ci
    var mp = ((rows + BM - 1) // BM) * BM
    var lp = ((lkv + BN - 1) // BN) * BN
    var out_shape = x.shape.copy()
    out_shape[len(out_shape) - 1] = out_lin.co
    var out = Tensor[F32](out_shape)

    var g = chain.g.copy()
    var s = g.scratch
    var a_need = mp * max(ci, out_lin.co)
    if s[].a_cap < a_need:
        s[].a = g.ctx.enqueue_create_buffer[F32](a_need)
        s[].a_cap = a_need

    _cross_pack_kv(g, k, v, h, d, lp)

    # upload x
    comptime NCHUNK = 16
    with s[].a.value().map_to_host() as hm:
        var ap = hm.unsafe_ptr()
        var xp = x.data.unsafe_ptr()
        var total = rows * ci
        var chunk = (total + NCHUNK - 1) // NCHUNK

        @parameter
        def copy_in(i: Int):
            var lo = i * chunk
            var n = min(chunk, total - lo)
            if n > 0:
                unsafe_memcpy(dest=ap.unsafe_offset(lo), src=xp.unsafe_offset(lo), count=n)

        parallelize[copy_in](NCHUNK)

    _cross_chain_enqueue(
        chain, g, s[].a.value(), rows, lkv, q_lin, out_lin, s[].a.value()
    )
    g.barrier()

    var op = out.data.unsafe_ptr()
    var co = out_lin.co
    var has_bias = out_lin.has_bias
    var bp = out_lin.bias_host.unsafe_ptr()
    with s[].a.value().map_to_host() as hm:
        var hp = hm.unsafe_ptr()
        var rchunk = (rows + NCHUNK - 1) // NCHUNK

        @parameter
        def copy_out(i: Int):
            var r0 = i * rchunk
            var r1 = min(r0 + rchunk, rows)
            if r1 <= r0:
                return
            if has_bias:
                comptime W = 8
                var co_main = (co // W) * W
                for r in range(r0, r1):
                    var base = r * co
                    var j = 0
                    while j < co_main:
                        op.unsafe_store(base + j, hp.unsafe_load[width=W](base + j) + bp.unsafe_load[width=W](j))
                        j += W
                    while j < co:
                        op[unsafe_offset=base + j] = hp[unsafe_offset=base + j] + bp[unsafe_offset=j]
                        j += 1
            else:
                unsafe_memcpy(dest=op.unsafe_offset(r0 * co), src=hp.unsafe_offset(r0 * co), count=(r1 - r0) * co)

        parallelize[copy_out](NCHUNK)
    return out^


def _cross_chain_enqueue(
    chain: GpuAttnChain,
    g: GpuContext,
    in_buf: DeviceBuffer[F32],
    rows: Int,
    lkv: Int,
    q_lin: GpuLinear,
    out_lin: GpuLinear,
    out_buf: DeviceBuffer[F32],
) raises:
    """Enqueue-only cross chain from a device-resident input [mp, C] to a
    device-resident output [mp, C]; reads the PRE-PACKED ckt/cvh (callers
    run _cross_pack_kv before any enqueue). The out-linear bias is NOT
    added (host wrapper: readback; block path: gate_add). No transfers,
    no barrier."""
    var h = chain.h
    var d = chain.d
    var hd = h * d
    var mp = ((rows + BM - 1) // BM) * BM
    var lp = ((lkv + BN - 1) // BN) * BN
    var s = g.scratch
    if s[].c_cap < mp * hd:
        s[].c = g.ctx.enqueue_create_buffer[F32](mp * hd)
        s[].c_cap = mp * hd
    var t = g.attn
    if t[].qh_cap < h * mp * d:
        t[].qh = g.ctx.enqueue_create_buffer[F32](h * mp * d)
        t[].qh_cap = h * mp * d
    var sc_need = _sdpa_sc_need(h, mp, lp)
    if t[].sc_cap < sc_need:
        t[].sc = g.ctx.enqueue_create_buffer[F32](sc_need)
        t[].sc_cap = sc_need
    if t[].ob_cap < h * mp * d:
        t[].ob = g.ctx.enqueue_create_buffer[F32](h * mp * d)
        t[].ob_cap = h * mp * d
    if t[].su_cap < h * mp:
        t[].su = g.ctx.enqueue_create_buffer[F32](h * mp)
        t[].su_cap = h * mp

    # q GEMM: c [mp, HD] = in [mp, C] @ Wq^T (dispatches on the weight
    # storage format — WP11 step 14)
    q_lin.enqueue_gemm(
        in_buf.unsafe_ptr(), s[].c.value().unsafe_ptr(), mp
    )
    # bias + q-rms on the valid rows, in place
    g.ctx.enqueue_function[bias_rms_q](
        rebind[Pointer[Scalar[F32], MutAnyOrigin]](s[].c.value().unsafe_ptr()),
        rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](chain.consts.unsafe_ptr()),
        Int32(rows), Int32(h), Int32(d), Int32(1 if chain.use_rms else 0),
        grid_dim=((rows * h + 255) // 256,), block_dim=(256,),
    )
    g.ctx.enqueue_function[pack_q_z](
        rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](s[].c.value().unsafe_ptr()),
        rebind[Pointer[Scalar[F32], MutAnyOrigin]](t[].qh.value().unsafe_ptr()),
        Int32(rows), Int32(h), Int32(d), Int32(mp), Int32(hd),
        grid_dim=((mp + 255) // 256, 1, h), block_dim=(256, 1, 1),
    )
    # sdpa composition against the pre-packed kv, in head groups (WP17)
    _enqueue_sdpa_groups(
        g,
        t[].qh.value().unsafe_ptr(), t[].ckt.value().unsafe_ptr(),
        t[].cvh.value().unsafe_ptr(), t[].sc.value().unsafe_ptr(),
        t[].su.value().unsafe_ptr(), t[].ob.value().unsafe_ptr(),
        h, mp, lp, lkv, d,
    )
    g.ctx.enqueue_function[unpack_o_z](
        rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](t[].ob.value().unsafe_ptr()),
        rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](t[].su.value().unsafe_ptr()),
        rebind[Pointer[Scalar[F32], MutAnyOrigin]](s[].c.value().unsafe_ptr()),
        Int32(mp), Int32(h), Int32(d),
        grid_dim=((mp + 255) // 256, 1, h), block_dim=(256, 1, 1),
    )
    out_lin.enqueue_gemm(
        s[].c.value().unsafe_ptr(), out_buf.unsafe_ptr(), mp
    )
