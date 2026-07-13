"""Reference side of the WP2 sampler parity test.

Runs the ORIGINAL FlowEuler samplers (plain / CFG / guidance-interval) with a
deterministic dummy velocity model. The Mojo driver replicates the model
natively and compares full trajectories.
"""

import os
import sys

os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from trellis2.pipelines.samplers.flow_euler import (  # noqa: E402
    FlowEulerSampler,
    FlowEulerCfgSampler,
    FlowEulerGuidanceIntervalSampler,
)

A, B, C = 0.7, 1.3, 0.9


class DummyModel:
    """v = sin(a*x + b*t) * c + cond — smooth, state- and t-dependent."""

    def __call__(self, x_t, t, cond):
        tt = (t.float() / 1000.0).view(-1, *([1] * (x_t.dim() - 1)))
        return torch.sin(x_t * A + tt * B) * C + cond


def gen(seed, n=2, c=3, d=5):
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(n, c, d, generator=g)
    cond = torch.randn(n, c, d, generator=g)
    neg_cond = torch.randn(n, c, d, generator=g)
    return noise, cond, neg_cond


def _pack(r):
    return {"samples": r.samples, "x_t": r.pred_x_t, "x_0": r.pred_x_0}


def run_plain(noise, cond, steps, rescale_t, sigma_min):
    s = FlowEulerSampler(sigma_min)
    return _pack(s.sample(DummyModel(), noise, cond, steps=steps, rescale_t=rescale_t, verbose=False))


def run_cfg(noise, cond, neg_cond, steps, rescale_t, sigma_min, strength):
    s = FlowEulerCfgSampler(sigma_min)
    return _pack(
        s.sample(
            DummyModel(), noise, cond, neg_cond,
            steps=steps, rescale_t=rescale_t, guidance_strength=strength, verbose=False,
        )
    )


def run_interval(noise, cond, neg_cond, steps, rescale_t, sigma_min, strength, lo, hi):
    s = FlowEulerGuidanceIntervalSampler(sigma_min)
    return _pack(
        s.sample(
            DummyModel(), noise, cond, neg_cond,
            steps=steps, rescale_t=rescale_t, guidance_strength=strength,
            guidance_interval=(lo, hi), verbose=False,
        )
    )


def run_interval_rescale(noise, cond, neg_cond, steps, rescale_t, sigma_min, strength, lo, hi, rescale):
    """Dense CFG rescale path (classifier_free_guidance_mixin) — the real
    ss-flow sampler config uses guidance_rescale 0.7."""
    s = FlowEulerGuidanceIntervalSampler(sigma_min)
    return _pack(
        s.sample(
            DummyModel(), noise, cond, neg_cond,
            steps=steps, rescale_t=rescale_t, guidance_strength=strength,
            guidance_interval=(lo, hi), guidance_rescale=rescale, verbose=False,
        )
    )


class VarlenDummyModel:
    """DummyModel math on SparseTensor feats. t is uniform across the batch
    during sampling, so the scalar form matches the Mojo flat-feats dummy."""

    def __call__(self, x_t, t, cond):
        tt = float(t.flatten()[0]) / 1000.0
        return x_t.replace(torch.sin(x_t.feats * A + tt * B) * C + cond)


def gen_varlen(seed, lengths, c=6):
    total = sum(lengths)
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(total, c, generator=g)
    cond = torch.randn(total, c, generator=g)
    neg_cond = torch.randn(total, c, generator=g)
    coords = torch.zeros(total, 4, dtype=torch.int32)
    row = 0
    for b, ln in enumerate(lengths):
        for i in range(ln):
            coords[row, 0] = b
            coords[row, 1] = i % 8
            coords[row, 2] = (i // 8) % 8
            coords[row, 3] = i // 64
            row += 1
    return feats, cond, neg_cond, coords


def run_interval_rescale_varlen(
    feats, cond, neg_cond, coords, steps, rescale_t, sigma_min, strength, lo, hi, rescale
):
    """Sparse CFG rescale path — VarLenTensor.std (biased, per segment) via
    the original SparseTensor, as the real shape-slat sampler (0.5) hits."""
    from trellis2.modules.sparse.basic import SparseTensor

    s = FlowEulerGuidanceIntervalSampler(sigma_min)
    r = s.sample(
        VarlenDummyModel(), SparseTensor(feats=feats, coords=coords), cond, neg_cond,
        steps=steps, rescale_t=rescale_t, guidance_strength=strength,
        guidance_interval=(lo, hi), guidance_rescale=rescale, verbose=False,
    )
    return {
        "samples": r.samples.feats,
        "x_t": [v.feats for v in r.pred_x_t],
        "x_0": [v.feats for v in r.pred_x_0],
    }
