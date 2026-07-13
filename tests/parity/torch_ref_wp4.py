"""Reference side of the WP4 parity test: linear, norms, nonlinearities,
modulate and RoPE — run through the ORIGINAL modules."""

import os
import sys

os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from trellis2.modules.sparse.basic import SparseTensor  # noqa: E402
from trellis2.modules.sparse.linear import SparseLinear  # noqa: E402
from trellis2.modules.sparse.attention.rope import SparseRotaryPositionEmbedder  # noqa: E402
from trellis2.modules.norm import LayerNorm32, GroupNorm32, ChannelLayerNorm32  # noqa: E402
from trellis2.modules.utils import modulate  # noqa: E402

from tests.parity.torch_ref import gen_case  # noqa: E402


def ref_linear(seed):
    g = torch.Generator().manual_seed(seed)
    feats, coords = gen_case(seed)
    ci = feats.shape[1]
    co = 7
    w = torch.randn(co, ci, generator=g)
    b = torch.randn(co, generator=g)
    lin = SparseLinear(ci, co, bias=True)
    with torch.no_grad():
        lin.weight.copy_(w)
        lin.bias.copy_(b)
    out = lin(SparseTensor(feats=feats, coords=coords))
    return {"feats": feats, "coords": coords, "w": w, "b": b, "out": out.feats}


def ref_layernorm(seed, affine):
    g = torch.Generator().manual_seed(seed)
    feats, _ = gen_case(seed)
    c = feats.shape[1]
    ln = LayerNorm32(c, elementwise_affine=affine, eps=1e-6)
    if affine:
        with torch.no_grad():
            ln.weight.copy_(torch.randn(c, generator=g))
            ln.bias.copy_(torch.randn(c, generator=g))
        w, b = ln.weight.data, ln.bias.data
    else:
        w = b = torch.zeros(c)
    return {"x": feats, "w": w, "b": b, "out": ln(feats)}


def ref_groupnorm(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 8, 3, 3, 3, generator=g)
    gn = GroupNorm32(4, 8)
    with torch.no_grad():
        gn.weight.copy_(torch.randn(8, generator=g))
        gn.bias.copy_(torch.randn(8, generator=g))
    return {"x": x, "w": gn.weight.data, "b": gn.bias.data, "out": gn(x)}


def ref_channel_layernorm(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 8, 3, 3, 3, generator=g)
    ln = ChannelLayerNorm32(8, eps=1e-6)
    with torch.no_grad():
        ln.weight.copy_(torch.randn(8, generator=g))
        ln.bias.copy_(torch.randn(8, generator=g))
    return {"x": x, "w": ln.weight.data, "b": ln.bias.data, "out": ln(x)}


def ref_nonlin(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(30, 5, generator=g) * 3
    return {
        "x": x,
        "relu": torch.nn.ReLU()(x),
        "silu": torch.nn.SiLU()(x),
        "gelu": torch.nn.GELU()(x),
    }


def ref_modulate(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 6, 4, generator=g)
    shift = torch.randn(2, 4, generator=g)
    scale = torch.randn(2, 4, generator=g)
    return {"x": x, "shift": shift, "scale": scale, "out": modulate(x, shift, scale)}


def ref_rope(seed, head_dim, heads=2):
    g = torch.Generator().manual_seed(seed)
    _, coords = gen_case(seed)
    n = coords.shape[0]
    qf = torch.randn(n, heads, head_dim, generator=g)
    kf = torch.randn(n, heads, head_dim, generator=g)
    q = SparseTensor(feats=qf, coords=coords)
    k = SparseTensor(feats=kf, coords=coords)
    rope = SparseRotaryPositionEmbedder(head_dim)
    q_e, k_e = rope(q, k)
    return {"coords": coords, "qf": qf, "kf": kf, "q_out": q_e.feats, "k_out": k_e.feats}
