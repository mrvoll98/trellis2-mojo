"""Reference side of the WP7 transformer-block parity tests.

Importing torch_ref_wp5 patches correct block-diagonal attention semantics
into the sparse attention module (the original's CPU fallback is buggy —
see that file's docstring); the blocks then compute reference outputs
through the ORIGINAL block classes with randomized weights.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from tests.parity.torch_ref_wp5 import gen_coords, correct_sdpa  # noqa: E402,F401  (patches sparse attn)

from trellis2.modules.sparse.basic import SparseTensor  # noqa: E402
from trellis2.modules.sparse.transformer.blocks import (  # noqa: E402
    SparseTransformerBlock,
    SparseTransformerCrossBlock,
)
from trellis2.modules.sparse.transformer.modulated import (  # noqa: E402
    ModulatedSparseTransformerBlock,
    ModulatedSparseTransformerCrossBlock,
)
from trellis2.modules.transformer.blocks import (  # noqa: E402
    AbsolutePositionEmbedder,
    TransformerBlock,
    TransformerCrossBlock,
)
from trellis2.modules.transformer.modulated import (  # noqa: E402
    ModulatedTransformerBlock,
    ModulatedTransformerCrossBlock,
)

H = 2
D = 8
C = H * D
CTX = 12
MLP_RATIO = 2.0


def _randomize(mod, seed):
    g = torch.Generator().manual_seed(seed + 5000)
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return mod


def _sd(block):
    return {k: v.detach() for k, v in block.state_dict().items()}


def _sparse_x(seed):
    g = torch.Generator().manual_seed(seed + 5100)
    coords = gen_coords(seed, 2, lo=2, hi=10)
    feats = torch.randn(coords.shape[0], C, generator=g)
    ctx = torch.randn(2, 5, CTX, generator=g)
    mod = torch.randn(2, C, generator=g)
    mod6 = torch.randn(2, 6 * C, generator=g)
    return feats, coords, ctx, mod, mod6


def ref_sparse_block(seed):
    feats, coords, _, _, _ = _sparse_x(seed)
    block = _randomize(SparseTransformerBlock(C, H, mlp_ratio=MLP_RATIO), seed)
    out = block(SparseTensor(feats=feats, coords=coords))
    return {"feats": feats, "coords": coords, "out": out.feats, "sd": _sd(block)}


def ref_sparse_cross_block(seed, ln_affine):
    feats, coords, ctx, _, _ = _sparse_x(seed)
    block = _randomize(
        SparseTransformerCrossBlock(C, CTX, H, mlp_ratio=MLP_RATIO, ln_affine=ln_affine), seed
    )
    out = block(SparseTensor(feats=feats, coords=coords), ctx)
    return {"feats": feats, "coords": coords, "ctx": ctx, "out": out.feats, "sd": _sd(block)}


def ref_mod_sparse_block(seed, share_mod):
    feats, coords, _, mod, mod6 = _sparse_x(seed)
    block = _randomize(
        ModulatedSparseTransformerBlock(C, H, mlp_ratio=MLP_RATIO, share_mod=share_mod), seed
    )
    m = mod6 if share_mod else mod
    out = block(SparseTensor(feats=feats, coords=coords), m)
    return {"feats": feats, "coords": coords, "mod": m, "out": out.feats, "sd": _sd(block)}


def ref_mod_sparse_cross_block(seed):
    feats, coords, ctx, mod, _ = _sparse_x(seed)
    block = _randomize(
        ModulatedSparseTransformerCrossBlock(C, CTX, H, mlp_ratio=MLP_RATIO, share_mod=False), seed
    )
    out = block(SparseTensor(feats=feats, coords=coords), mod, ctx)
    return {"feats": feats, "coords": coords, "ctx": ctx, "mod": mod, "out": out.feats, "sd": _sd(block)}


def _dense_x(seed, n=2, l=6):
    g = torch.Generator().manual_seed(seed + 5200)
    x = torch.randn(n, l, C, generator=g)
    ctx = torch.randn(n, 5, CTX, generator=g)
    mod = torch.randn(n, C, generator=g)
    mod6 = torch.randn(n, 6 * C, generator=g)
    return x, ctx, mod, mod6


def ref_dense_block(seed):
    x, _, _, _ = _dense_x(seed)
    block = _randomize(TransformerBlock(C, H, mlp_ratio=MLP_RATIO), seed)
    return {"x": x, "out": block(x), "sd": _sd(block)}


def ref_dense_cross_block(seed):
    x, ctx, _, _ = _dense_x(seed)
    block = _randomize(TransformerCrossBlock(C, CTX, H, mlp_ratio=MLP_RATIO, ln_affine=True), seed)
    return {"x": x, "ctx": ctx, "out": block(x, ctx), "sd": _sd(block)}


def ref_mod_dense_block(seed, share_mod):
    x, _, mod, mod6 = _dense_x(seed)
    block = _randomize(ModulatedTransformerBlock(C, H, mlp_ratio=MLP_RATIO, share_mod=share_mod), seed)
    m = mod6 if share_mod else mod
    return {"x": x, "mod": m, "out": block(x, m), "sd": _sd(block)}


def ref_mod_dense_cross_block(seed):
    x, ctx, mod, _ = _dense_x(seed)
    block = _randomize(
        ModulatedTransformerCrossBlock(C, CTX, H, mlp_ratio=MLP_RATIO, share_mod=False), seed
    )
    return {"x": x, "ctx": ctx, "mod": mod, "out": block(x, mod, ctx), "sd": _sd(block)}


def ref_abs_pos(seed, channels=16):
    g = torch.Generator().manual_seed(seed + 5300)
    pos = torch.randint(0, 64, (7, 3), generator=g).float()
    emb = AbsolutePositionEmbedder(channels)
    return {"pos": pos, "out": emb(pos)}
