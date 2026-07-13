"""Reference side of the WP5 attention parity tests.

BUG FOUND: the original's naive/sdpa fallback in
modules/sparse/attention/full_attn.py zero-pads k/v to the max sequence
length WITHOUT masking, so queries attend to padding whenever batches have
unequal lengths (max diff ~1.8 vs correct semantics on random cases). The
production backends (flash_attn/xformers varlen) are block-diagonal and
correct. The Mojo port implements the correct semantics, so the reference
here is `correct_sdpa` (per-batch torch sdpa) — patched into the original
SparseMultiHeadAttention for the glue tests.

Windowed attention has NO CPU backend at all in the original (only
xformers/flash_attn); `windowed_ref` replicates its semantics per window.
"""

import math
import os
import sys

os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
os.environ.setdefault("ATTN_BACKEND", "naive")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from trellis2.modules.sparse.basic import SparseTensor  # noqa: E402
from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention  # noqa: E402
from trellis2.modules.sparse.attention import modules as sp_attn_modules  # noqa: E402
from trellis2.modules.attention.full_attn import scaled_dot_product_attention  # noqa: E402
from trellis2.modules.attention.modules import MultiHeadAttention  # noqa: E402

from tests.parity.torch_ref import gen_case  # noqa: E402

H = 2
D = 8
C = H * D


def gen_coords(seed, b, grid=6, lo=1, hi=9):
    g = torch.Generator().manual_seed(seed)
    coords = []
    for i in range(b):
        n = int(torch.randint(lo, hi, (1,), generator=g))
        flat = torch.randperm(grid**3, generator=g)[:n]
        x = flat // (grid * grid)
        y = (flat // grid) % grid
        z = flat % grid
        coords.append(torch.stack([torch.full((n,), i), x, y, z], dim=1))
    return torch.cat(coords, dim=0).int()


def _sdpa_block(qs, ks, vs):
    return F.scaled_dot_product_attention(
        qs.permute(1, 0, 2), ks.permute(1, 0, 2), vs.permute(1, 0, 2)
    ).permute(1, 0, 2)


def correct_sdpa(*args):
    """Block-diagonal varlen attention with per-batch torch sdpa — the
    semantics the flash_attn/xformers backends implement."""
    if len(args) == 1:
        qkv = args[0]
        out = torch.empty(qkv.feats.shape[0], qkv.feats.shape[2], qkv.feats.shape[3])
        for b in range(qkv.shape[0]):
            q, k, v = qkv.feats[qkv.layout[b]].unbind(1)
            out[qkv.layout[b]] = _sdpa_block(q, k, v)
        return qkv.replace(out)
    if len(args) == 2:
        q, kv = args
        out = torch.empty(q.feats.shape[0], q.feats.shape[1], kv.shape[-1])
        for b in range(q.shape[0]):
            if isinstance(kv, torch.Tensor):
                k, v = kv[b].unbind(1)
            else:
                k, v = kv.feats[kv.layout[b]].unbind(1)
            out[q.layout[b]] = _sdpa_block(q.feats[q.layout[b]], k, v)
        return q.replace(out)
    q, k, v = args
    out = torch.empty(q.feats.shape[0], q.feats.shape[1], v.shape[-1])
    for b in range(q.shape[0]):
        if isinstance(k, torch.Tensor):
            ks, vs = k[b], v[b]
        else:
            ks, vs = k.feats[k.layout[b]], v.feats[v.layout[b]]
        out[q.layout[b]] = _sdpa_block(q.feats[q.layout[b]], ks, vs)
    return q.replace(out)


# the MHA glue must also run on correct attention semantics
sp_attn_modules.sparse_scaled_dot_product_attention = correct_sdpa


def ref_full_qkv(seed):
    g = torch.Generator().manual_seed(seed + 100)
    coords = gen_coords(seed, 2)
    qkv = torch.randn(coords.shape[0], 3, H, D, generator=g)
    x = SparseTensor(feats=qkv, coords=coords)
    return {"coords": coords, "qkv": qkv, "out": correct_sdpa(x).feats}


def ref_q_kv_sparse(seed):
    g = torch.Generator().manual_seed(seed + 200)
    qc = gen_coords(seed, 2)
    kvc = gen_coords(seed + 1, 2)
    qf = torch.randn(qc.shape[0], H, D, generator=g)
    kvf = torch.randn(kvc.shape[0], 2, H, D, generator=g)
    q = SparseTensor(feats=qf, coords=qc)
    kv = SparseTensor(feats=kvf, coords=kvc)
    out = correct_sdpa(q, kv)
    return {"qc": qc, "kvc": kvc, "qf": qf, "kvf": kvf, "out": out.feats}


def ref_q_kv_dense(seed, l=5):
    g = torch.Generator().manual_seed(seed + 300)
    qc = gen_coords(seed, 2)
    qf = torch.randn(qc.shape[0], H, D, generator=g)
    kv = torch.randn(2, l, 2, H, D, generator=g)
    q = SparseTensor(feats=qf, coords=qc)
    out = correct_sdpa(q, kv)
    return {"qc": qc, "qf": qf, "kv": kv, "out": out.feats}


def ref_q_k_v_sparse(seed, co=6):
    g = torch.Generator().manual_seed(seed + 400)
    qc = gen_coords(seed, 2)
    kvc = gen_coords(seed + 1, 2)
    qf = torch.randn(qc.shape[0], H, D, generator=g)
    kf = torch.randn(kvc.shape[0], H, D, generator=g)
    vf = torch.randn(kvc.shape[0], H, co, generator=g)
    out = correct_sdpa(
        SparseTensor(feats=qf, coords=qc),
        SparseTensor(feats=kf, coords=kvc),
        SparseTensor(feats=vf, coords=kvc),
    )
    return {"qc": qc, "kvc": kvc, "qf": qf, "kf": kf, "vf": vf, "out": out.feats}


def ref_dense_sdpa(seed, n=2, l=6, co=6):
    g = torch.Generator().manual_seed(seed + 500)
    qkv = torch.randn(n, l, 3, H, D, generator=g)
    q = torch.randn(n, l, H, D, generator=g)
    kv = torch.randn(n, l + 2, 2, H, D, generator=g)
    k = torch.randn(n, l + 1, H, D, generator=g)
    v = torch.randn(n, l + 1, H, co, generator=g)
    return {
        "qkv": qkv, "q": q, "kv": kv, "k": k, "v": v,
        "out_qkv": scaled_dot_product_attention(qkv),
        "out_q_kv": scaled_dot_product_attention(q, kv),
        "out_q_k_v": scaled_dot_product_attention(q, k, v),
    }


def windowed_ref(x, window_size, shift_window=(0, 0, 0)):
    """Exact semantics of sparse_windowed_scaled_dot_product_self_attention,
    on CPU. x: SparseTensor with feats [T, 3, H, C]."""
    coords = x.coords
    dim = coords.shape[1] - 1
    ws = (window_size,) * dim if isinstance(window_size, int) else window_size
    sp = [int(v) for v in (coords[:, 1:].max(0).values + 1)]
    num_w = [math.ceil((s + sh + 1) / w) for s, sh, w in zip(sp, shift_window, ws)]
    keys = coords[:, 0].long()
    for i in range(dim):
        w = (coords[:, i + 1].long() + shift_window[i]) // ws[i]
        keys = keys * num_w[i] + w
    fwd = torch.argsort(keys, stable=True)
    feats = x.feats[fwd]
    sorted_keys = keys[fwd]
    out_sorted = torch.empty(feats.shape[0], feats.shape[2], feats.shape[3])
    start = 0
    for i in range(1, feats.shape[0] + 1):
        if i == feats.shape[0] or sorted_keys[i] != sorted_keys[start]:
            q, k, v = feats[start:i].unbind(1)  # [m, H, C]
            o = F.scaled_dot_product_attention(
                q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
            ).permute(1, 0, 2)
            out_sorted[start:i] = o
            start = i
    bwd = torch.empty_like(fwd)
    bwd[fwd] = torch.arange(fwd.shape[0])
    return x.replace(out_sorted[bwd])


def ref_windowed(seed, window_size, shift):
    g = torch.Generator().manual_seed(seed + 600)
    coords = gen_coords(seed, 2, grid=6, lo=4, hi=14)
    qkv = torch.randn(coords.shape[0], 3, H, D, generator=g)
    x = SparseTensor(feats=qkv, coords=coords)
    out = windowed_ref(x, window_size, tuple(shift))
    return {"coords": coords, "qkv": qkv, "out": out.feats}


def _mha_weights(mha):
    w = {"to_out_w": mha.to_out.weight.data, "to_out_b": mha.to_out.bias.data}
    if hasattr(mha, "to_qkv"):
        w["to_qkv_w"] = mha.to_qkv.weight.data
        w["to_qkv_b"] = mha.to_qkv.bias.data
    if hasattr(mha, "to_q"):
        w["to_q_w"] = mha.to_q.weight.data
        w["to_q_b"] = mha.to_q.bias.data
        w["to_kv_w"] = mha.to_kv.weight.data
        w["to_kv_b"] = mha.to_kv.bias.data
    if getattr(mha, "qk_rms_norm", False):
        w["gamma_q"] = mha.q_rms_norm.gamma.data
        w["gamma_k"] = mha.k_rms_norm.gamma.data
    return w


def _randomize(mha, seed):
    g = torch.Generator().manual_seed(seed + 700)
    with torch.no_grad():
        for p in mha.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return mha


def ref_sparse_mha_self(seed, qk_rms_norm, use_rope, attn_mode="full", window_size=0):
    mha = sp_attn_modules.SparseMultiHeadAttention(
        C, H, attn_mode=attn_mode, window_size=window_size or None,
        shift_window=(0, 0, 0), use_rope=use_rope, qk_rms_norm=qk_rms_norm,
    )
    _randomize(mha, seed)
    g = torch.Generator().manual_seed(seed + 800)
    coords = gen_coords(seed, 2, lo=2, hi=10)
    feats = torch.randn(coords.shape[0], C, generator=g)
    x = SparseTensor(feats=feats, coords=coords)
    out = mha(x)
    return {"coords": coords, "feats": feats, "out": out.feats, **_mha_weights(mha)}


def ref_sparse_mha_double_windowed(seed, window_size=2):
    # the original windowed kernel cannot run on CPU: patch in windowed_ref
    orig = sp_attn_modules.sparse_windowed_scaled_dot_product_self_attention
    sp_attn_modules.sparse_windowed_scaled_dot_product_self_attention = (
        lambda qkv, ws, shift_window=(0, 0, 0): windowed_ref(qkv, ws, tuple(shift_window))
    )
    try:
        return ref_sparse_mha_self(seed, False, False, "double_windowed", window_size)
    finally:
        sp_attn_modules.sparse_windowed_scaled_dot_product_self_attention = orig


def ref_sparse_mha_cross(seed, qk_rms_norm, ctx_channels=12, l=5):
    mha = sp_attn_modules.SparseMultiHeadAttention(
        C, H, ctx_channels=ctx_channels, type="cross", qk_rms_norm=qk_rms_norm
    )
    _randomize(mha, seed)
    g = torch.Generator().manual_seed(seed + 900)
    coords = gen_coords(seed, 2, lo=2, hi=10)
    feats = torch.randn(coords.shape[0], C, generator=g)
    ctx = torch.randn(2, l, ctx_channels, generator=g)
    out = mha(SparseTensor(feats=feats, coords=coords), ctx)
    return {"coords": coords, "feats": feats, "ctx": ctx, "out": out.feats, **_mha_weights(mha)}


def ref_dense_mha(seed, type, qk_rms_norm, n=2, l=6, ctx_channels=12, lkv=5):
    mha = MultiHeadAttention(
        C, H, ctx_channels=ctx_channels if type == "cross" else None,
        type=type, qk_rms_norm=qk_rms_norm,
    )
    _randomize(mha, seed)
    g = torch.Generator().manual_seed(seed + 1000)
    x = torch.randn(n, l, C, generator=g)
    ctx = torch.randn(n, lkv, ctx_channels, generator=g)
    out = mha(x, ctx) if type == "cross" else mha(x)
    return {"x": x, "ctx": ctx, "out": out, **_mha_weights(mha)}
