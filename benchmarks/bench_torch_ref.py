"""Torch baseline side of the WP10 benchmarks (driven by bench_wp10.mojo).

Case generators (seeded, parametric sizes) plus timed runners for the hot
inference paths. Semantics of the baselines:

- Sparse full attention has TWO torch numbers: `padded` is the original's
  naive CPU fallback (vectorized but INCORRECT — attends to zero-padding
  when batch lengths differ, see tests/parity/torch_ref_wp5.py), `correct`
  is the per-batch torch-SDPA loop that matches what the production
  flash/xformers backends compute (and what the Mojo port implements).
- Windowed attention has no CPU backend upstream at all; the baseline is
  the per-window torch-SDPA reference from the parity tests.
- Blocks and models run with the correct attention patched in (the
  torch_ref_wp5 import does that), matching the parity-test setup.

Timing: warmup runs first (which also fills the layout/neighbor caches on
both sides — steady-state hot-path numbers), then `iters` timed runs under
torch.no_grad(); the list of per-run seconds is returned and the Mojo side
reports the minimum. Set BENCH_TORCH_THREADS to pin torch's thread count.
"""

import os
import sys
import time
import platform

os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
os.environ.setdefault("ATTN_BACKEND", "naive")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

if os.environ.get("BENCH_TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["BENCH_TORCH_THREADS"]))

from tests.parity.torch_ref_wp5 import correct_sdpa, windowed_ref  # noqa: E402  (patches sparse attn)

from trellis2.modules.sparse.basic import SparseTensor  # noqa: E402
from trellis2.modules.sparse.attention.full_attn import (  # noqa: E402
    sparse_scaled_dot_product_attention,
)
from trellis2.modules.sparse.conv.conv import SparseConv3d  # noqa: E402
from trellis2.modules.sparse.transformer.modulated import (  # noqa: E402
    ModulatedSparseTransformerBlock,
)
from trellis2.models.structured_latent_flow import SLatFlowModel  # noqa: E402
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler  # noqa: E402

RESCALE_T = 3.0
CFG = 5.0
INTERVAL = (0.2, 0.9)
SIGMA_MIN = 1e-5


def machine_info():
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
    }


def bench_coords(seed, tokens, batches, grid):
    """`tokens` unique voxels per batch inside a grid^3 volume."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for b in range(batches):
        flat = torch.randperm(grid**3, generator=g)[:tokens]
        x = flat // (grid * grid)
        y = (flat // grid) % grid
        z = flat % grid
        rows.append(torch.stack([torch.full((tokens,), b), x, y, z], dim=1))
    return torch.cat(rows, dim=0).int()


def _randomize(mod, seed):
    g = torch.Generator().manual_seed(seed + 7000)
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return mod


def _time(fn, iters, warmup=1):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        out = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            out.append(time.perf_counter() - t0)
    return out


# -- sparse full self-attention (qkv packed) ---------------------------------

def case_attn(seed, tokens, batches, heads, head_dim, grid):
    g = torch.Generator().manual_seed(seed + 1)
    coords = bench_coords(seed, tokens, batches, grid)
    qkv = torch.randn(coords.shape[0], 3, heads, head_dim, generator=g)
    return {"coords": coords, "qkv": qkv}


def time_attn_padded(case, iters):
    x = SparseTensor(feats=case["qkv"], coords=case["coords"])
    return _time(lambda: sparse_scaled_dot_product_attention(x), iters)


def time_attn_correct(case, iters):
    x = SparseTensor(feats=case["qkv"], coords=case["coords"])
    return _time(lambda: correct_sdpa(x), iters)


def time_attn_windowed(case, window, iters):
    x = SparseTensor(feats=case["qkv"], coords=case["coords"])
    return _time(lambda: windowed_ref(x, window), iters)


# -- submanifold sparse conv (conv_none backend) ------------------------------

def case_conv(seed, tokens, batches, cin, cout, ks, grid):
    g = torch.Generator().manual_seed(seed + 2)
    coords = bench_coords(seed, tokens, batches, grid)
    feats = torch.randn(coords.shape[0], cin, generator=g)
    conv = SparseConv3d(cin, cout, ks)
    with torch.no_grad():
        conv.weight.copy_(torch.randn(conv.weight.shape, generator=g) * 0.3)
        conv.bias.copy_(torch.randn(conv.bias.shape, generator=g))
    return {
        "coords": coords, "feats": feats,
        "w": conv.weight.data, "b": conv.bias.data,
        "_conv": conv,
    }


def time_conv(case, iters):
    x = SparseTensor(feats=case["feats"], coords=case["coords"])
    conv = case["_conv"]
    return _time(lambda: conv(x), iters)


# -- modulated sparse transformer block (share_mod, like real checkpoints) ----

def case_block(seed, tokens, batches, ch, heads, grid, mlp_ratio=4.0):
    g = torch.Generator().manual_seed(seed + 3)
    coords = bench_coords(seed, tokens, batches, grid)
    feats = torch.randn(coords.shape[0], ch, generator=g)
    mod = torch.randn(batches, 6 * ch, generator=g)
    block = _randomize(
        ModulatedSparseTransformerBlock(ch, heads, mlp_ratio=mlp_ratio, share_mod=True),
        seed,
    )
    return {
        "coords": coords, "feats": feats, "mod": mod,
        "sd": {k: v.detach() for k, v in block.state_dict().items()},
        "_block": block,
    }


def time_block(case, iters):
    x = SparseTensor(feats=case["feats"], coords=case["coords"])
    block, mod = case["_block"], case["mod"]
    return _time(lambda: block(x, mod), iters)


# -- SLat flow sampling loop (FlowEuler + CFG-interval, whole trajectory) -----

def case_sampler(seed, tokens, batches, ch, model_ch, blocks, heads,
                 cond_len, cond_ch, steps, grid):
    g = torch.Generator().manual_seed(seed + 4)
    coords = bench_coords(seed, tokens, batches, grid)
    noise = torch.randn(coords.shape[0], ch, generator=g)
    cond = torch.randn(batches, cond_len, cond_ch, generator=g)
    model = _randomize(SLatFlowModel(
        resolution=grid, in_channels=ch, model_channels=model_ch,
        cond_channels=cond_ch, out_channels=ch, num_blocks=blocks,
        num_heads=heads, mlp_ratio=4.0, pe_mode="rope", share_mod=True,
        qk_rms_norm=True, qk_rms_norm_cross=True,
    ), seed)
    return {
        "coords": coords, "noise": noise, "cond": cond, "steps": steps,
        "sd": {k: v.detach() for k, v in model.state_dict().items()},
        "_model": model,
    }


def time_sampler(case, iters):
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=SIGMA_MIN)
    x = SparseTensor(feats=case["noise"], coords=case["coords"])
    cond, neg_cond = case["cond"], torch.zeros_like(case["cond"])
    model, steps = case["_model"], case["steps"]

    def run():
        sampler.sample(
            model, x, cond, neg_cond, steps=steps, rescale_t=RESCALE_T,
            guidance_strength=CFG, guidance_interval=INTERVAL, verbose=False,
        )

    return _time(run, iters)
