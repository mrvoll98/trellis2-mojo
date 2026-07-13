"""Reference side of the WP9 pipeline-core integration test.

Replicates the model-facing stages of Trellis2ImageTo3DPipeline with the
ORIGINAL torch samplers and models (small, randomized weights):
sparse-structure sampling -> decode -> occupancy coords (full and pooled
resolution), shape-SLat sampling on those coords -> de-normalization ->
sparse UNet decode -> FlexiDualGrid head transforms.

Thresholding (occupancy > 0, intersected > 0) can flip on tiny numeric
drift between implementations, so the raw pre-threshold tensors are
exported: the Mojo side compares those numerically, verifies the
coords/threshold logic exactly on the torch tensors, and treats
intersection-bit mismatches as failures only when the torch logit is not
borderline.

Importing torch_ref_wp5 patches the sparse-attention CPU fallback (the SLat
flow model needs it).
"""

import os
import sys

os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
os.environ.setdefault("ATTN_BACKEND", "naive")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from tests.parity.torch_ref_wp5 import correct_sdpa  # noqa: E402,F401  (patches sparse attn)

from trellis2.modules.sparse.basic import SparseTensor  # noqa: E402
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel  # noqa: E402
from trellis2.models.sparse_structure_vae import SparseStructureDecoder  # noqa: E402
from trellis2.models.structured_latent_flow import SLatFlowModel  # noqa: E402
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder  # noqa: E402
from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler  # noqa: E402

N = 2
COND_L = 5
COND_C = 12
SS_RESO = 4
SS_CH = 4
SLAT_CH = 4
STEPS = 5
RESCALE_T = 3.0
CFG = 5.0
INTERVAL = (0.2, 0.9)
GR_SS = 0.7    # real pipeline.json: ss guidance_rescale (dense std)
GR_SLAT = 0.5  # real pipeline.json: shape-slat guidance_rescale (varlen std)
SIGMA_MIN = 1e-5


def _randomize(mod, seed):
    g = torch.Generator().manual_seed(seed + 9000)
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return mod


def _run(seed):
    g = torch.Generator().manual_seed(seed + 9100)
    cond = torch.randn(N, COND_L, COND_C, generator=g)
    neg_cond = torch.zeros_like(cond)  # pipeline get_cond semantics

    ss_flow = _randomize(SparseStructureFlowModel(
        resolution=SS_RESO, in_channels=SS_CH, model_channels=24, cond_channels=COND_C,
        out_channels=SS_CH, num_blocks=2, num_heads=2, mlp_ratio=2.0, pe_mode="rope",
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
    ), seed)
    ss_dec = _randomize(SparseStructureDecoder(
        out_channels=1, latent_channels=SS_CH, num_res_blocks=1, channels=[8, 4],
        num_res_blocks_middle=1, norm_type="layer",
    ), seed + 1)
    slat_flow = _randomize(SLatFlowModel(
        resolution=16, in_channels=SLAT_CH, model_channels=24, cond_channels=COND_C,
        out_channels=SLAT_CH, num_blocks=2, num_heads=2, mlp_ratio=2.0, pe_mode="rope",
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
    ), seed + 2)
    unet_dec = _randomize(SparseUnetVaeDecoder(
        out_channels=7, model_channels=[16, 8], latent_channels=SLAT_CH,
        num_blocks=[1, 1],
        block_type=["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
        up_block_type=["SparseResBlockC2S3d"],
        block_args=[{}, {}], pred_subdiv=True,
    ), seed + 3).eval()

    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=SIGMA_MIN)

    # -- sample_sparse_structure
    noise = torch.randn(N, SS_CH, SS_RESO, SS_RESO, SS_RESO, generator=g)
    with torch.no_grad():
        z = sampler.sample(
            ss_flow, noise, cond, neg_cond, steps=STEPS, rescale_t=RESCALE_T,
            guidance_strength=CFG, guidance_interval=INTERVAL,
            guidance_rescale=GR_SS, verbose=False,
        ).samples
        occ_raw = ss_dec(z)  # [N, 1, 8, 8, 8]
    decoded = occ_raw > 0
    coords8 = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    pooled = torch.nn.functional.max_pool3d(decoded.float(), 2, 2, 0) > 0.5
    coords4 = torch.argwhere(pooled)[:, [0, 2, 3, 4]].int()
    if coords4.shape[0] == 0 or torch.any(coords4[:, 0].bincount(minlength=N) == 0):
        return None  # a batch with no active voxels — pick another seed

    # -- sample_shape_slat on the pooled coords
    noise2 = torch.randn(coords4.shape[0], SLAT_CH, generator=g)
    with torch.no_grad():
        slat = sampler.sample(
            slat_flow, SparseTensor(feats=noise2, coords=coords4), cond, neg_cond,
            steps=STEPS, rescale_t=RESCALE_T, guidance_strength=CFG,
            guidance_interval=INTERVAL, guidance_rescale=GR_SLAT, verbose=False,
        ).samples
    std = torch.rand(SLAT_CH, generator=g) * 0.5 + 0.75
    mean = torch.randn(SLAT_CH, generator=g) * 0.2
    slat = slat * std[None] + mean[None]

    # -- decode_shape_slat core (mesh extraction stays in Python glue)
    with torch.no_grad():
        out, subs = unet_dec(slat, return_subs=True)
    margin = 0.5
    vertices = (1 + 2 * margin) * torch.sigmoid(out.feats[:, 0:3]) - margin
    intersected = (out.feats[:, 3:6] > 0).float()
    quad_lerp = torch.nn.functional.softplus(out.feats[:, 6:7])

    # -- sample_shape_slat_cascade: LR slat (reuses slat above) -> decoder
    # upsample -> quantize/unique -> HR slat with a second flow model
    LR_RESOLUTION = 16
    CASCADE_RESOLUTION = 64
    hr_flow = _randomize(SLatFlowModel(
        resolution=16, in_channels=SLAT_CH, model_channels=24, cond_channels=COND_C,
        out_channels=SLAT_CH, num_blocks=2, num_heads=2, mlp_ratio=2.0, pe_mode="rope",
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
    ), seed + 4)
    with torch.no_grad():
        hr_coords = unet_dec.upsample(slat, upsample_times=1)
    k = CASCADE_RESOLUTION // 16
    quant = torch.cat([
        hr_coords[:, :1],
        ((hr_coords[:, 1:] + 0.5) / LR_RESOLUTION * k).int(),
    ], dim=1)
    cascade_c = quant.unique(dim=0).int()
    noise3 = torch.randn(cascade_c.shape[0], SLAT_CH, generator=g)
    with torch.no_grad():
        hr_slat = sampler.sample(
            hr_flow, SparseTensor(feats=noise3, coords=cascade_c), cond, neg_cond,
            steps=STEPS, rescale_t=RESCALE_T, guidance_strength=CFG,
            guidance_interval=INTERVAL, guidance_rescale=GR_SLAT, verbose=False,
        ).samples
    hr_slat = hr_slat * std[None] + mean[None]

    # -- texture stage: sample_tex_slat (concat of the normalized shape
    # slat) + decode_tex_slat (guided by the shape decoder's subs)
    TEX_CH = 3
    tex_flow = _randomize(SLatFlowModel(
        resolution=16, in_channels=SLAT_CH + TEX_CH, model_channels=24,
        cond_channels=COND_C, out_channels=TEX_CH, num_blocks=2, num_heads=2,
        mlp_ratio=2.0, pe_mode="rope", share_mod=True, qk_rms_norm=True,
        qk_rms_norm_cross=True,
    ), seed + 5)
    tex_dec = _randomize(SparseUnetVaeDecoder(
        out_channels=6, model_channels=[16, 8], latent_channels=TEX_CH,
        num_blocks=[1, 1],
        block_type=["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
        up_block_type=["SparseResBlockC2S3d"],
        block_args=[{}, {}], pred_subdiv=False,
    ), seed + 6).eval()
    tex_std = torch.rand(TEX_CH, generator=g) * 0.5 + 0.75
    tex_mean = torch.randn(TEX_CH, generator=g) * 0.2
    shape_slat_norm = (slat - mean[None]) / std[None]
    noise4 = torch.randn(slat.feats.shape[0], TEX_CH, generator=g)
    with torch.no_grad():
        tex_slat = sampler.sample(
            tex_flow, slat.replace(noise4), cond, neg_cond,
            concat_cond=shape_slat_norm,
            steps=STEPS, rescale_t=RESCALE_T, guidance_strength=CFG,
            guidance_interval=INTERVAL, guidance_rescale=GR_SLAT, verbose=False,
        ).samples
        tex_slat = tex_slat * tex_std[None] + tex_mean[None]
        tex_out = tex_dec(tex_slat, guide_subs=subs) * 0.5 + 0.5

    return {
        "cond": cond,
        "ss_sd": {k: v.detach() for k, v in ss_flow.state_dict().items()},
        "ss_dec_sd": {k: v.detach() for k, v in ss_dec.state_dict().items()},
        "slat_sd": {k: v.detach() for k, v in slat_flow.state_dict().items()},
        "unet_sd": {k: v.detach() for k, v in unet_dec.state_dict().items()},
        "noise": noise,
        "coords8": coords8,
        "coords4": coords4,
        "occ_raw": occ_raw,
        "noise2": noise2,
        "mean": mean,
        "std": std,
        "slat": slat.feats,
        "out_feats": out.feats,
        "out_coords": out.coords,
        "sub0": subs[0].feats,
        "vertices": vertices,
        "intersected": intersected,
        "quad_lerp": quad_lerp,
        "lr_resolution": LR_RESOLUTION,
        "cascade_resolution": CASCADE_RESOLUTION,
        "hr_sd": {k: v.detach() for k, v in hr_flow.state_dict().items()},
        "hr_coords": hr_coords,
        "cascade_coords": cascade_c,
        "noise3": noise3,
        "hr_slat": hr_slat.feats,
        "tex_sd": {k: v.detach() for k, v in tex_flow.state_dict().items()},
        "tex_dec_sd": {k: v.detach() for k, v in tex_dec.state_dict().items()},
        "tex_mean": tex_mean,
        "tex_std": tex_std,
        "noise4": noise4,
        "tex_slat": tex_slat.feats,
        "tex_out": tex_out.feats,
        "tex_out_coords": tex_out.coords,
    }


def ref_pipeline_core(seed):
    """Retry seed offsets only if a batch ends up with no active voxels."""
    for offset in range(0, 1000, 100):
        r = _run(seed + offset)
        if r is not None:
            r["seed_offset"] = offset
            return r
    raise RuntimeError("no seed offset with active voxels in every batch")
