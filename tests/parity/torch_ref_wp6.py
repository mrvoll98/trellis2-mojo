"""Reference side of the WP6 parity tests: SparseDownsample/Upsample,
SparseSpatial2Channel/Channel2Spatial and the conv_none SparseConv3d —
all through the ORIGINAL modules (these run fine on CPU)."""

import os
import sys

os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
os.environ.setdefault("ATTN_BACKEND", "naive")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from trellis2.modules.sparse.basic import SparseTensor  # noqa: E402
from trellis2.modules.sparse.spatial.basic import SparseDownsample, SparseUpsample  # noqa: E402
from trellis2.modules.sparse.spatial.spatial2channel import (  # noqa: E402
    SparseSpatial2Channel,
    SparseChannel2Spatial,
)
from trellis2.modules.sparse.conv.conv import SparseConv3d  # noqa: E402

from tests.parity.torch_ref_wp5 import gen_coords  # noqa: E402

C = 8


def _case(seed, lo=4, hi=14):
    g = torch.Generator().manual_seed(seed + 42)
    coords = gen_coords(seed, 2, grid=6, lo=lo, hi=hi)
    feats = torch.randn(coords.shape[0], C, generator=g)
    return feats, coords


def ref_down_up(seed, mode):
    feats, coords = _case(seed)
    x = SparseTensor(feats=feats, coords=coords)
    down = SparseDownsample(2, mode)
    down.train(False)
    y = down(x)
    z = SparseUpsample(2)(y)
    assert torch.equal(z.coords, coords)
    return {
        "feats": feats, "coords": coords,
        "down_feats": y.feats, "down_coords": y.coords,
        "up_feats": z.feats,
    }


def ref_up_subdivision(seed):
    g = torch.Generator().manual_seed(seed + 43)
    feats, coords = _case(seed, lo=2, hi=8)
    x = SparseTensor(feats=feats, coords=coords)
    sub = torch.zeros(coords.shape[0], 8, dtype=torch.bool)
    sub[torch.arange(coords.shape[0]), torch.randint(0, 8, (coords.shape[0],), generator=g)] = True
    extra = torch.rand(coords.shape[0], 8, generator=g) < 0.3
    sub |= extra
    subdivision = SparseTensor(feats=sub, coords=coords)
    y = SparseUpsample(2)(x, subdivision=subdivision)
    return {"feats": feats, "coords": coords, "sub": sub.float(), "out_feats": y.feats, "out_coords": y.coords}


def ref_s2c_c2s(seed):
    feats, coords = _case(seed)
    x = SparseTensor(feats=feats, coords=coords)
    s2c = SparseSpatial2Channel(2)
    s2c.train(False)
    y = s2c(x)
    z = SparseChannel2Spatial(2)(y)
    assert torch.equal(z.coords, coords)
    assert torch.equal(z.feats, feats)  # exact roundtrip
    return {
        "feats": feats, "coords": coords,
        "s2c_feats": y.feats, "s2c_coords": y.coords,
        "c2s_feats": z.feats,
    }


def ref_conv(seed, ks, dilation):
    g = torch.Generator().manual_seed(seed + 44)
    feats, coords = _case(seed)
    x = SparseTensor(feats=feats, coords=coords)
    conv = SparseConv3d(C, 5, ks, dilation=dilation)
    with torch.no_grad():
        conv.weight.copy_(torch.randn(conv.weight.shape, generator=g) * 0.3)
        conv.bias.copy_(torch.randn(conv.bias.shape, generator=g))
    out = conv(x)
    out2 = conv(x)  # second run through the neighbor cache
    assert torch.equal(out.feats, out2.feats)
    return {
        "feats": feats, "coords": coords,
        "w": conv.weight.data, "b": conv.bias.data,
        "out": out.feats,
    }
