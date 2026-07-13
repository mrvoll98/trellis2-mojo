# WP11 step 3: GPU dense SDPA (GEMM composition, gpu/attention.mojo) vs
# the CPU varlen_sdpa via dense_sdpa_q_k_v. GPU exp + accumulation order
# differ from both CPU paths -> tolerance parity, never bit-equality
# (pass 8 precedent). Also checks the kv-padding/mask path (odd context
# lengths), the shape gate, and the MultiHeadAttention dispatch wiring.
# WP11 step 7 adds the chained-path cases: whole-MHA parity (qkv->bias/
# rms/rope->sdpa->out device-resident) for dense plain, dense rms+rope,
# and sparse rms+rope (phases from coords), plus chain build/gate checks.
# Chain cases need C = 1024: smaller weights stay under GPU_MIN_WEIGHT.
#
# Runs on the Metal GPU; SKIPS loudly if no device is available.

from trellis2_mojo.gpu.context import GpuContext
from trellis2_mojo.gpu.attention import gpu_dense_sdpa, gpu_varlen_sdpa_single, gpu_sdpa_wants
from trellis2_mojo.gpu.block import (
    gpu_block_phases,
    gpu_block_state_readback,
    gpu_block_state_upload,
)
from trellis2_mojo.io.state_dict import StateDict
from trellis2_mojo.loaders import dense_mha_from, sparse_mha_from, dense_ffn_from, sparse_ffn_from
from trellis2_mojo.modules.nn import LayerNorm32
from trellis2_mojo.modules.rope import RotaryPositionEmbedder
from trellis2_mojo.modules.transformer.modulated import ModulatedTransformerCrossBlock
from trellis2_mojo.sparse.attention.full_attn import dense_sdpa_q_k_v, varlen_sdpa
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix
from trellis2_mojo.sparse.transformer.modulated import (
    Modulation,
    ModulatedSparseTransformerCrossBlock,
)

comptime F32 = DType.float32


def fill(mut t: Tensor[F32], seed: Int, scale: Float32):
    var state = seed * 6364136223846793005 + 1442695040888963407
    for i in range(len(t.data)):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32((state >> 33) & ((1 << 20) - 1)) / Float32(1 << 19)
        t.data[i] = (u - 1.0) * scale


def max_diff(a: Tensor[F32], b: Tensor[F32]) raises -> Float32:
    if len(a.data) != len(b.data):
        raise Error("shape mismatch in max_diff")
    var m: Float32 = 0
    for i in range(len(a.data)):
        var d = abs(a.data[i] - b.data[i])
        if d > m:
            m = d
    return m


def block_sd(c: Int, h: Int, ffn: Int, seed: Int) raises -> StateDict:
    """State dict for a full cross-block (self/cross attention + mlp) +
    the norm2 affine params and the share_mod modulation vector."""
    var d = Dict[String, Tensor[F32]]()
    var wq = Tensor[F32]([3 * c, c])
    var bq = Tensor[F32]([3 * c])
    var wo = Tensor[F32]([c, c])
    var bo = Tensor[F32]([c])
    fill(wq, seed, 0.05)
    fill(bq, seed + 1, 0.1)
    fill(wo, seed + 2, 0.05)
    fill(bo, seed + 3, 0.1)
    d["self_attn.to_qkv.weight"] = wq^
    d["self_attn.to_qkv.bias"] = bq^
    d["self_attn.to_out.weight"] = wo^
    d["self_attn.to_out.bias"] = bo^
    var cwq = Tensor[F32]([c, c])
    var cbq = Tensor[F32]([c])
    var cwkv = Tensor[F32]([2 * c, c])
    var cbkv = Tensor[F32]([2 * c])
    var cwo = Tensor[F32]([c, c])
    var cbo = Tensor[F32]([c])
    fill(cwq, seed + 4, 0.05)
    fill(cbq, seed + 5, 0.1)
    fill(cwkv, seed + 6, 0.05)
    fill(cbkv, seed + 7, 0.1)
    fill(cwo, seed + 8, 0.05)
    fill(cbo, seed + 9, 0.1)
    d["cross_attn.to_q.weight"] = cwq^
    d["cross_attn.to_q.bias"] = cbq^
    d["cross_attn.to_kv.weight"] = cwkv^
    d["cross_attn.to_kv.bias"] = cbkv^
    d["cross_attn.to_out.weight"] = cwo^
    d["cross_attn.to_out.bias"] = cbo^
    var wf1 = Tensor[F32]([ffn, c])
    var bf1 = Tensor[F32]([ffn])
    var wf2 = Tensor[F32]([c, ffn])
    var bf2 = Tensor[F32]([c])
    fill(wf1, seed + 10, 0.05)
    fill(bf1, seed + 11, 0.1)
    fill(wf2, seed + 12, 0.05)
    fill(bf2, seed + 13, 0.1)
    d["mlp.mlp.0.weight"] = wf1^
    d["mlp.mlp.0.bias"] = bf1^
    d["mlp.mlp.2.weight"] = wf2^
    d["mlp.mlp.2.bias"] = bf2^
    var idx = 14
    for name in ["self_attn.q_rms_norm.gamma", "self_attn.k_rms_norm.gamma",
                 "cross_attn.q_rms_norm.gamma", "cross_attn.k_rms_norm.gamma"]:
        var gm = Tensor[F32]([h, c // h])
        fill(gm, seed + idx, 0.25)
        for i in range(len(gm.data)):
            gm.data[i] += 1.0
        d[String(name)] = gm^
        idx += 1
    var n2w = Tensor[F32]([c])
    var n2b = Tensor[F32]([c])
    fill(n2w, seed + idx, 0.25)
    fill(n2b, seed + idx + 1, 0.1)
    for i in range(c):
        n2w.data[i] += 1.0
    d["norm2.weight"] = n2w^
    d["norm2.bias"] = n2b^
    var md = Tensor[F32]([6 * c])
    fill(md, seed + idx + 2, 0.2)
    d["modulation"] = md^
    return StateDict(d^)


def dense_block_from(sd: StateDict, c: Int, h: Int) raises -> ModulatedTransformerCrossBlock:
    var n2 = LayerNorm32(c, 1e-6, True)
    n2.weight = sd.tensor("norm2.weight")
    n2.bias = sd.tensor("norm2.bias")
    return ModulatedTransformerCrossBlock(
        c,
        LayerNorm32(c, 1e-6, False), n2^, LayerNorm32(c, 1e-6, False),
        dense_mha_from(sd, "self_attn", c, h, qk_rms_norm=True),
        dense_mha_from(sd, "cross_attn", c, h, is_cross=True, qk_rms_norm=True),
        dense_ffn_from(sd, "mlp"),
        Modulation(sd.tensor("modulation")),
    )


def sparse_block_from(sd: StateDict, c: Int, h: Int) raises -> ModulatedSparseTransformerCrossBlock:
    var n2 = LayerNorm32(c, 1e-6, True)
    n2.weight = sd.tensor("norm2.weight")
    n2.bias = sd.tensor("norm2.bias")
    return ModulatedSparseTransformerCrossBlock(
        c,
        LayerNorm32(c, 1e-6, False), n2^, LayerNorm32(c, 1e-6, False),
        sparse_mha_from(sd, "self_attn", c, h, use_rope=True, qk_rms_norm=True),
        sparse_mha_from(sd, "cross_attn", c, h, is_cross=True, qk_rms_norm=True),
        sparse_ffn_from(sd, "mlp"),
        Modulation(sd.tensor("modulation")),
    )


def chain_sd(c: Int, h: Int, seed: Int, qk_rms: Bool) raises -> StateDict:
    """State dict for a self-attention MHA at C channels / H heads; gammas
    near 1 when qk_rms (a zero-centered gamma would be degenerate)."""
    var d = Dict[String, Tensor[F32]]()
    var wq = Tensor[F32]([3 * c, c])
    var bq = Tensor[F32]([3 * c])
    var wo = Tensor[F32]([c, c])
    var bo = Tensor[F32]([c])
    fill(wq, seed, 0.05)
    fill(bq, seed + 1, 0.1)
    fill(wo, seed + 2, 0.05)
    fill(bo, seed + 3, 0.1)
    d["attn.to_qkv.weight"] = wq^
    d["attn.to_qkv.bias"] = bq^
    d["attn.to_out.weight"] = wo^
    d["attn.to_out.bias"] = bo^
    if qk_rms:
        var gq = Tensor[F32]([h, c // h])
        var gk = Tensor[F32]([h, c // h])
        fill(gq, seed + 4, 0.25)
        fill(gk, seed + 5, 0.25)
        for i in range(len(gq.data)):
            gq.data[i] += 1.0
            gk.data[i] += 1.0
        d["attn.q_rms_norm.gamma"] = gq^
        d["attn.k_rms_norm.gamma"] = gk^
    return StateDict(d^)


def cross_sd(c: Int, h: Int, cctx: Int, seed: Int, qk_rms: Bool) raises -> StateDict:
    """State dict for a cross-attention MHA (to_q/to_kv/to_out)."""
    var d = Dict[String, Tensor[F32]]()
    var wq = Tensor[F32]([c, c])
    var bq = Tensor[F32]([c])
    var wkv = Tensor[F32]([2 * c, cctx])
    var bkv = Tensor[F32]([2 * c])
    var wo = Tensor[F32]([c, c])
    var bo = Tensor[F32]([c])
    fill(wq, seed, 0.05)
    fill(bq, seed + 1, 0.1)
    fill(wkv, seed + 2, 0.05)
    fill(bkv, seed + 3, 0.1)
    fill(wo, seed + 4, 0.05)
    fill(bo, seed + 5, 0.1)
    d["attn.to_q.weight"] = wq^
    d["attn.to_q.bias"] = bq^
    d["attn.to_kv.weight"] = wkv^
    d["attn.to_kv.bias"] = bkv^
    d["attn.to_out.weight"] = wo^
    d["attn.to_out.bias"] = bo^
    if qk_rms:
        var gq = Tensor[F32]([h, c // h])
        var gk = Tensor[F32]([h, c // h])
        fill(gq, seed + 6, 0.25)
        fill(gk, seed + 7, 0.25)
        for i in range(len(gq.data)):
            gq.data[i] += 1.0
            gk.data[i] += 1.0
        d["attn.q_rms_norm.gamma"] = gq^
        d["attn.k_rms_norm.gamma"] = gk^
    return StateDict(d^)


def check_case(
    gpu: GpuContext, l: Int, lkv: Int, h: Int, d: Int, seed: Int, atol: Float32
) raises:
    var q = Tensor[F32]([1, l, h, d])
    var k = Tensor[F32]([1, lkv, h, d])
    var v = Tensor[F32]([1, lkv, h, d])
    fill(q, seed, 1.0)
    fill(k, seed + 1, 1.0)
    fill(v, seed + 2, 1.0)
    var cpu = dense_sdpa_q_k_v(q, k, v)
    var dev = gpu_dense_sdpa(gpu, q, k, v)
    var diff = max_diff(cpu, dev)
    print("  gpu vs cpu sdpa  L", l, " Lkv", lkv, " H", h, " D", d, " max|diff|:", diff)
    if diff > atol:
        raise Error("gpu sdpa parity failed")


def check_single(
    gpu: GpuContext, t: Int, tkv: Int, h: Int, d: Int, seed: Int, atol: Float32
) raises:
    """gpu_varlen_sdpa_single (odd lengths, q-padding — WP11 step 4) vs
    the CPU varlen_sdpa with single-segment offsets."""
    var q = Tensor[F32]([t, h, d])
    var k = Tensor[F32]([tkv, h, d])
    var v = Tensor[F32]([tkv, h, d])
    fill(q, seed, 1.0)
    fill(k, seed + 1, 1.0)
    fill(v, seed + 2, 1.0)
    var qo: List[Int] = [0, t]
    var ko: List[Int] = [0, tkv]
    var cpu = varlen_sdpa(q, k, v, qo, ko)
    var dev = gpu_varlen_sdpa_single(gpu, q, k, v)
    var diff = max_diff(cpu, dev)
    print("  gpu vs cpu single-seg  T", t, " Tkv", tkv, " H", h, " D", d, " max|diff|:", diff)
    if diff > atol:
        raise Error("gpu single-segment sdpa parity failed")


def main() raises:
    var gpu: GpuContext
    try:
        gpu = GpuContext()
    except:
        print("wp11 gpu attn: SKIPPED — no usable GPU device")
        return

    # self-attention at DiT scale (CPU takes the flash path at kv >= 1024)
    check_case(gpu, 2048, 2048, 4, 64, 61, 5e-5)
    # cross-attention with an odd context length (kv padding + mask)
    check_case(gpu, 2048, 1029, 4, 64, 62, 5e-5)
    # kv below the CPU flash threshold (exact CPU path as reference)
    check_case(gpu, 2048, 999, 4, 64, 63, 5e-5)
    # more heads, DiT-real geometry at reduced length
    check_case(gpu, 4096, 4096, 2, 64, 64, 5e-5)
    # WP17: head-grouped composition — 16 heads x 4160^2 scores exceed
    # GPU_SDPA_MAX_SCORES as one buffer (previously DECLINED) but fit per
    # head, so this enqueues in 15 + 1 head groups against the shared
    # scores scratch (the 1024-cascade HR-slat geometry class)
    check_case(gpu, 4160, 4160, 16, 64, 68, 5e-5)

    # step 4: odd lengths on BOTH sides (q-padding) — the B=1 slat cases
    check_single(gpu, 2369, 2369, 4, 64, 65, 5e-5)   # slat self
    check_single(gpu, 2369, 1029, 4, 64, 66, 5e-5)   # slat cross
    check_single(gpu, 2113, 999, 4, 64, 67, 5e-5)    # odd + sub-flash kv

    # WP19 trinn 2: f16-shared sdpa-GEMM-er (TRELLIS2_GPU_F16) — samme
    # komposisjon med gemm_z_f16sh; kildene er aktiveringer så utgangen
    # er IKKE bit-eksakt (f16-cast på begge shared-fyll, ~5e-4 rel inn
    # i logitene) men innenfor den løsere toleranseklassen
    var g16 = gpu.copy()
    g16.f16 = True
    check_case(g16, 2048, 2048, 4, 64, 61, 5e-3)
    check_single(g16, 2369, 1029, 4, 64, 66, 5e-3)
    print("  f16-shared sdpa: opt-in via kontekst-flagget, av som default")

    # shape gate (q %64 is no longer required — padding handles it)
    if not gpu_sdpa_wants(2369, 2369, 64, 4):
        raise Error("wants rejected the odd slat self shape")
    if gpu_sdpa_wants(512, 2048, 64, 4):
        raise Error("wants accepted L below the floor")
    if gpu_sdpa_wants(2048, 128, 64, 4):
        raise Error("wants accepted tiny kv")
    if gpu_sdpa_wants(2048, 2048, 48, 4):
        raise Error("wants accepted D % 64 != 0")
    if not gpu_sdpa_wants(4096, 1029, 64, 16):
        raise Error("wants rejected the ss_flow cross shape")

    # MultiHeadAttention dispatch wiring: dense_mha_from with sd.gpu set
    # must give the same result (within tolerance) as the CPU instance
    comptime C = 256
    comptime H = 4
    var l = 2048
    var d = Dict[String, Tensor[F32]]()
    var wq = Tensor[F32]([3 * C, C])
    var bq = Tensor[F32]([3 * C])
    var wo = Tensor[F32]([C, C])
    var bo = Tensor[F32]([C])
    fill(wq, 71, 0.05)
    fill(bq, 72, 0.1)
    fill(wo, 73, 0.05)
    fill(bo, 74, 0.1)
    d["attn.to_qkv.weight"] = wq^
    d["attn.to_qkv.bias"] = bq^
    d["attn.to_out.weight"] = wo^
    d["attn.to_out.bias"] = bo^
    var sd = StateDict(d^)
    var mha_cpu = dense_mha_from(sd, "attn", C, H)
    sd.gpu = gpu.copy()
    var mha_gpu = dense_mha_from(sd, "attn", C, H)
    if not mha_gpu.gpu:
        raise Error("dense_mha_from did not attach the GPU context")
    var x = Tensor[F32]([1, l, C])
    fill(x, 75, 1.0)
    var diff = max_diff(mha_cpu.forward(x), mha_gpu.forward(x))
    print("  MultiHeadAttention dispatch (L 2048, C 256, H 4)  max|diff|:", diff)
    if diff > 2e-4:
        raise Error("MHA dispatch parity failed")

    # SparseMultiHeadAttention dispatch (step 4): single-segment sparse
    # self-attention through sparse_mha_from with and without sd.gpu
    var t = 2369
    var ds = Dict[String, Tensor[F32]]()
    var swq = Tensor[F32]([3 * C, C])
    var sbq = Tensor[F32]([3 * C])
    var swo = Tensor[F32]([C, C])
    var sbo = Tensor[F32]([C])
    fill(swq, 81, 0.05)
    fill(sbq, 82, 0.1)
    fill(swo, 83, 0.05)
    fill(sbo, 84, 0.1)
    ds["attn.to_qkv.weight"] = swq^
    ds["attn.to_qkv.bias"] = sbq^
    ds["attn.to_out.weight"] = swo^
    ds["attn.to_out.bias"] = sbo^
    var ssd = StateDict(ds^)
    var smha_cpu = sparse_mha_from(ssd, "attn", C, H)
    ssd.gpu = gpu.copy()
    var smha_gpu = sparse_mha_from(ssd, "attn", C, H)
    if not smha_gpu.gpu:
        raise Error("sparse_mha_from did not attach the GPU context")
    var feats = Tensor[F32]([t, C])
    fill(feats, 85, 1.0)
    var coords = IntMatrix(t, 4)
    for r in range(t):
        coords.set(r, 1, r % 32)
        coords.set(r, 2, (r // 32) % 32)
        coords.set(r, 3, r // 1024)
    var sx = SparseTensor[F32](feats.copy(), coords.copy(), 1)
    var sdiff = max_diff(
        smha_cpu.forward(sx).vl.feats, smha_gpu.forward(sx).vl.feats
    )
    print("  SparseMultiHeadAttention dispatch (T 2369, C 256, H 4)  max|diff|:", sdiff)
    if sdiff > 2e-4:
        raise Error("sparse MHA dispatch parity failed")

    # ---- WP11 step 7: chained self-attention (device-resident) ----
    comptime CC = 1024
    comptime CH = 16

    # dense, plain (no rms/rope): the chain must be built AND dispatched
    var csd = chain_sd(CC, CH, 91, False)
    var cmha_cpu = dense_mha_from(csd, "attn", CC, CH)
    csd.gpu = gpu.copy()
    var cmha_gpu = dense_mha_from(csd, "attn", CC, CH)
    if not cmha_gpu.chain:
        raise Error("dense_mha_from did not build the attention chain")
    if not cmha_gpu.chain.value().wants(2048, cmha_gpu.to_qkv.gpu.value()):
        raise Error("chain gate rejected the dense chain shape")
    if cmha_gpu.chain.value().wants(512, cmha_gpu.to_qkv.gpu.value()):
        raise Error("chain gate accepted L below the sdpa floor")
    var cx = Tensor[F32]([1, 2048, CC])
    fill(cx, 92, 1.0)
    var cdiff = max_diff(cmha_cpu.forward(cx), cmha_gpu.forward(cx))
    print("  chained dense plain (L 2048, C 1024, H 16)  max|diff|:", cdiff)
    if cdiff > 5e-4:
        raise Error("chained dense plain parity failed")

    # dense, rms + rope (the ss_flow configuration)
    var rsd = chain_sd(CC, CH, 101, True)
    var rmha_cpu = dense_mha_from(rsd, "attn", CC, CH, qk_rms_norm=True)
    rsd.gpu = gpu.copy()
    var rmha_gpu = dense_mha_from(rsd, "attn", CC, CH, qk_rms_norm=True)
    if not rmha_gpu.chain:
        raise Error("dense_mha_from (rms) did not build the attention chain")
    var emb = RotaryPositionEmbedder(CC // CH)
    var pos = Tensor[F32]([2048, 3])
    for r in range(2048):
        pos.data[r * 3] = Float32(r % 16)
        pos.data[r * 3 + 1] = Float32((r // 16) % 16)
        pos.data[r * 3 + 2] = Float32(r // 256)
    var phases = emb.forward(pos)
    var rx = Tensor[F32]([1, 2048, CC])
    fill(rx, 102, 1.0)
    var rdiff = max_diff(rmha_cpu.forward(rx, phases), rmha_gpu.forward(rx, phases))
    print("  chained dense rms+rope (L 2048, C 1024, H 16)  max|diff|:", rdiff)
    if rdiff > 5e-4:
        raise Error("chained dense rms+rope parity failed")

    # sparse, rms + rope from coords (the slat DiT configuration)
    var ssd7 = chain_sd(CC, CH, 111, True)
    var smha7_cpu = sparse_mha_from(ssd7, "attn", CC, CH, use_rope=True, qk_rms_norm=True)
    ssd7.gpu = gpu.copy()
    var smha7_gpu = sparse_mha_from(ssd7, "attn", CC, CH, use_rope=True, qk_rms_norm=True)
    if not smha7_gpu.chain:
        raise Error("sparse_mha_from did not build the attention chain")
    var sfeats = Tensor[F32]([t, CC])
    fill(sfeats, 112, 1.0)
    var sx7 = SparseTensor[F32](sfeats.copy(), coords.copy(), 1)
    var s7diff = max_diff(
        smha7_cpu.forward(sx7).vl.feats, smha7_gpu.forward(sx7).vl.feats
    )
    print("  chained sparse rms+rope (T 2369, C 1024, H 16)  max|diff|:", s7diff)
    if s7diff > 5e-4:
        raise Error("chained sparse rms+rope parity failed")

    # ---- WP11 step 8: chained cross-attention (q side device-resident) ----

    # dense cross, rms (the ss_flow cross configuration): also checks the
    # self/cross build declines (cross MHA: no self chain; and vice versa)
    var xsd = cross_sd(CC, CH, CC, 131, True)
    var xmha_cpu = dense_mha_from(xsd, "attn", CC, CH, is_cross=True, qk_rms_norm=True)
    xsd.gpu = gpu.copy()
    var xmha_gpu = dense_mha_from(xsd, "attn", CC, CH, is_cross=True, qk_rms_norm=True)
    if xmha_gpu.chain:
        raise Error("cross MHA built a self-attention chain")
    if not xmha_gpu.cross_chain:
        raise Error("dense_mha_from (cross) did not build the cross chain")
    if cmha_gpu.cross_chain:
        raise Error("self MHA built a cross chain")
    if not xmha_gpu.cross_chain.value().wants_cross(2048, 1029):
        raise Error("cross gate rejected the ss cross shape")
    if xmha_gpu.cross_chain.value().wants_cross(2048, 128):
        raise Error("cross gate accepted tiny kv")
    if xmha_gpu.cross_chain.value().wants_cross(512, 1029):
        raise Error("cross gate accepted L below the floor")
    var xq = Tensor[F32]([1, 2048, CC])
    var xctx = Tensor[F32]([1, 1029, CC])
    fill(xq, 132, 1.0)
    fill(xctx, 133, 1.0)
    var xdiff = max_diff(
        xmha_cpu.forward_cross(xq, xctx), xmha_gpu.forward_cross(xq, xctx)
    )
    print("  chained dense cross rms (L 2048 x 1029, C 1024, H 16)  max|diff|:", xdiff)
    if xdiff > 5e-4:
        raise Error("chained dense cross parity failed")

    # dense cross, plain (no rms) — dense_sdpa_q_kv CPU reference path
    var psd = cross_sd(CC, CH, CC, 141, False)
    var pmha_cpu = dense_mha_from(psd, "attn", CC, CH, is_cross=True)
    psd.gpu = gpu.copy()
    var pmha_gpu = dense_mha_from(psd, "attn", CC, CH, is_cross=True)
    if not pmha_gpu.cross_chain:
        raise Error("dense_mha_from (cross plain) did not build the cross chain")
    var pdiff = max_diff(
        pmha_cpu.forward_cross(xq, xctx), pmha_gpu.forward_cross(xq, xctx)
    )
    print("  chained dense cross plain (L 2048 x 1029, C 1024, H 16)  max|diff|:", pdiff)
    if pdiff > 5e-4:
        raise Error("chained dense cross plain parity failed")

    # sparse cross, rms (the slat cross configuration; T = 2369 odd)
    var scsd = cross_sd(CC, CH, CC, 151, True)
    var scmha_cpu = sparse_mha_from(scsd, "attn", CC, CH, is_cross=True, qk_rms_norm=True)
    scsd.gpu = gpu.copy()
    var scmha_gpu = sparse_mha_from(scsd, "attn", CC, CH, is_cross=True, qk_rms_norm=True)
    if not scmha_gpu.cross_chain:
        raise Error("sparse_mha_from (cross) did not build the cross chain")
    var scfeats = Tensor[F32]([t, CC])
    fill(scfeats, 152, 1.0)
    var scx = SparseTensor[F32](scfeats.copy(), coords.copy(), 1)
    var scctx = Tensor[F32]([1, 1029, CC])
    fill(scctx, 153, 1.0)
    var scdiff = max_diff(
        scmha_cpu.forward_cross(scx, scctx).vl.feats,
        scmha_gpu.forward_cross(scx, scctx).vl.feats,
    )
    print("  chained sparse cross rms (T 2369 x 1029, C 1024, H 16)  max|diff|:", scdiff)
    if scdiff > 5e-4:
        raise Error("chained sparse cross parity failed")

    # ---- WP11 step 10: whole-block residency ----

    # dense cross-block (rms+rope self, rms cross, share_mod)
    var bsd = block_sd(CC, CH, 4 * CC, 161)
    var blk_cpu = dense_block_from(bsd, CC, CH)
    bsd.gpu = gpu.copy()
    var blk_gpu = dense_block_from(bsd, CC, CH)
    if not blk_gpu._gpu_block_ok(1, 2048, 1029):
        raise Error("dense block gate rejected the qualifying shape")
    if blk_gpu._gpu_block_ok(1, 512, 1029):
        raise Error("dense block gate accepted L below the floor")
    var bx = Tensor[F32]([1, 2048, CC])
    var bmod = Tensor[F32]([1, 6 * CC])
    fill(bx, 162, 1.0)
    fill(bmod, 163, 0.2)
    var bdiff = max_diff(
        blk_cpu.forward(bx, bmod, xctx, phases),
        blk_gpu.forward(bx, bmod, xctx, phases),
    )
    print("  whole dense block (L 2048 x 1029, C 1024, H 16)  max|diff|:", bdiff)
    if bdiff > 1e-3:
        raise Error("whole dense block parity failed")

    # sparse cross-block (rope from coords)
    var sbsd = block_sd(CC, CH, 4 * CC, 171)
    var sblk_cpu = sparse_block_from(sbsd, CC, CH)
    sbsd.gpu = gpu.copy()
    var sblk_gpu = sparse_block_from(sbsd, CC, CH)
    var sbfeats = Tensor[F32]([t, CC])
    fill(sbfeats, 172, 1.0)
    var sbx = SparseTensor[F32](sbfeats.copy(), coords.copy(), 1)
    if not sblk_gpu._gpu_block_ok(sbx, xctx):
        raise Error("sparse block gate rejected the qualifying shape")
    var sbmod = Tensor[F32]([1, 6 * CC])
    fill(sbmod, 173, 0.2)
    var sbdiff = max_diff(
        sblk_cpu.forward(sbx, sbmod, xctx).vl.feats,
        sblk_gpu.forward(sbx, sbmod, xctx).vl.feats,
    )
    print("  whole sparse block (T 2369 x 1029, C 1024, H 16)  max|diff|:", sbdiff)
    if sbdiff > 1e-3:
        raise Error("whole sparse block parity failed")

    # ---- WP11 step 12: model-level residency ----
    # the resident 2-block driver must be BIT-identical to the sequential
    # per-block GPU path: the readback/upload it drops was an exact copy
    var b2sd = block_sd(CC, CH, 4 * CC, 181)
    b2sd.gpu = gpu.copy()
    var blk_a = dense_block_from(b2sd, CC, CH)
    var b3sd = block_sd(CC, CH, 4 * CC, 191)
    b3sd.gpu = gpu.copy()
    var blk_b = dense_block_from(b3sd, CC, CH)
    var rx2 = Tensor[F32]([1, 2048, CC])
    fill(rx2, 182, 1.0)
    var rmod = Tensor[F32]([1, 6 * CC])
    fill(rmod, 183, 0.2)
    var seq = blk_b.forward(blk_a.forward(rx2, rmod, xctx, phases), rmod, xctx, phases)
    var g2 = blk_a.self_attn.chain.value().g.copy()
    var ph2 = gpu_block_phases(g2, phases, 2048, CC // CH)
    gpu_block_state_upload(g2, rx2, 2048, CC)
    blk_a._gpu_enqueue_resident(rmod, xctx, True, ph2, 2048)
    blk_b._gpu_enqueue_resident(rmod, xctx, True, ph2, 2048)
    g2.barrier()
    var res = gpu_block_state_readback(g2, rx2.shape, 2048, CC)
    var rdiff2 = max_diff(seq, res)
    print("  model-resident 2-block vs sequential  max|diff|:", rdiff2)
    if rdiff2 != 0:
        raise Error("model-resident path is not bit-identical to per-block")

    print(
        "wp11 gpu-sdpa parity vs cpu: 7 shapes + gate + dense/sparse MHA"
        " dispatch + 3 chained self + 3 chained cross + 2 whole-block"
        " + resident-driver cases + chain gates passed"
    )
