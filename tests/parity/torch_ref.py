"""Reference side of the WP3 parity fuzz.

Runs the ORIGINAL PyTorch implementation (trellis2/modules/sparse/basic.py)
on seeded random cases. The Mojo driver (parity_sparse_vs_torch.mojo) imports
this module through Python interop, feeds identical data through the Mojo
port, and compares results.
"""

import os
import sys

os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from trellis2.modules.sparse.basic import (  # noqa: E402
    SparseTensor,
    VarLenTensor,
    sparse_cat,
)


def gen_case(seed, grid=8):
    """Random batch of unique voxel coords (batch-contiguous) + features."""
    g = torch.Generator().manual_seed(seed)
    B = int(torch.randint(1, 5, (1,), generator=g))
    C = int(torch.randint(1, 9, (1,), generator=g))
    coords = []
    for b in range(B):
        n = int(torch.randint(1, 13, (1,), generator=g))
        flat = torch.randperm(grid**3, generator=g)[:n]
        x = flat // (grid * grid)
        y = (flat // grid) % grid
        z = flat % grid
        coords.append(torch.stack([torch.full((n,), b), x, y, z], dim=1))
    coords = torch.cat(coords, dim=0).int()
    feats = torch.randn(coords.shape[0], C, generator=g)
    return feats, coords


def dense_ref(x):
    """Intended semantics of SparseTensor.to_dense() for the 'none' backend.

    The original (basic.py:687) crashes on this path with
    `list + tuple` — `.unbind(1)` returns a tuple. Bug found by this parity
    test; reimplemented here with the obvious fix.
    """
    ret = torch.zeros(*x.shape, *x.spatial_shape, dtype=x.dtype, device=x.device)
    idx = [x.coords[:, 0].long(), slice(None)] + list(x.coords[:, 1:].long().unbind(1))
    ret[tuple(idx)] = x.feats
    return ret


def ref_results(feats, coords, seed):
    g = torch.Generator().manual_seed(seed + 10000)
    x = SparseTensor(feats=feats, coords=coords)
    B = x.shape[0]
    C = feats.shape[1]

    other_feats = torch.randn(feats.shape, generator=g)
    other = x.replace(other_feats)
    batch_full = torch.randn(B, C, generator=g)
    batch_one = torch.randn(B, 1, generator=g)
    perm = torch.randperm(B, generator=g).tolist()

    feats2, coords2 = gen_case(seed + 20000)
    feats2 = feats2[:, :1].expand(-1, C).contiguous()  # match channel count
    x2 = SparseTensor(feats=feats2, coords=coords2)

    y = x[perm]
    z0 = sparse_cat([x, x2], dim=0)
    z1 = sparse_cat([x, other], dim=1)

    vl = VarLenTensor(feats, layout=x.layout)
    vld, vlm = vl.to_dense()

    res = {
        "other_feats": other_feats,
        "batch_full": batch_full,
        "batch_one": batch_one,
        "perm": perm,
        "feats2": feats2,
        "coords2": coords2,
        "add": (x + other).feats,
        "mul_s": (x * 2.5).feats,
        "rsub": (1.5 - x).feats,
        "div_s": (x / 1.7).feats,
        "neg": (-x).feats,
        "badd_full": (x + batch_full).feats,
        "bmul_one": (x * batch_one).feats,
        "get_feats": y.feats,
        "get_coords": y.coords,
        "get_stops": [s.stop for s in y.layout],
        "cat0_feats": z0.feats,
        "cat0_coords": z0.coords,
        "cat1_feats": z1.feats,
        "dense": dense_ref(x),
        "vl_dense": vld,
        "vl_mask": vlm.int(),
        "mean_b": x.mean(dim=1),
        "sum_b": x.sum(dim=1),
    }
    return res


def ref_full():
    x = SparseTensor.full([0, 0, 0, 2, 1, 3], (2, 3), 0.5)
    return {"coords": x.coords, "feats": x.feats}
