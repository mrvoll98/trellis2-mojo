"""Reference side of the WP8 model parity tests (sparse_structure_flow +
sparse_structure_vae decoder + structured_latent_flow).

The SLat flow model runs sparse attention, so importing torch_ref_wp5
below patches the original's buggy CPU sdpa fallback with correct
block-diagonal semantics (see that file's docstring). The two dense models
don't need it. Weights are randomized and exposed via state_dict,
mirroring the other torch_ref files.
"""

import os
import sys

os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
os.environ.setdefault("ATTN_BACKEND", "naive")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from tests.parity.torch_ref_wp5 import gen_coords, correct_sdpa  # noqa: E402,F401  (patches sparse attn)

from trellis2.modules.sparse.basic import SparseTensor  # noqa: E402
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel  # noqa: E402
from trellis2.models.sparse_structure_vae import ResBlock3d, SparseStructureDecoder  # noqa: E402
from trellis2.models.structured_latent_flow import SLatFlowModel  # noqa: E402
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder  # noqa: E402

R = 4
IN_C = 4
OUT_C = 4
COND_C = 12
COND_L = 5
NUM_BLOCKS = 2
MLP_RATIO = 2.0
N = 2


def _randomize(mod, seed):
    g = torch.Generator().manual_seed(seed + 8000)
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return mod


def _inputs(seed):
    g = torch.Generator().manual_seed(seed + 8100)
    x = torch.randn(N, IN_C, R, R, R, generator=g)
    t = torch.rand(N, generator=g) * 1000
    cond = torch.randn(N, COND_L, COND_C, generator=g)
    return x, t, cond


def ref_resblock3d(seed, cin, cout, norm_type):
    g = torch.Generator().manual_seed(seed + 8200)
    x = torch.randn(N, cin, 3, 3, 3, generator=g)
    block = _randomize(ResBlock3d(cin, cout, norm_type=norm_type), seed)
    with torch.no_grad():
        out = block(x)
    return {"x": x, "out": out, "sd": {k: v.detach() for k, v in block.state_dict().items()}}


def ref_ss_decoder(seed, norm_type):
    # GroupNorm32 hardcodes 32 groups, so the group case needs channels % 32 == 0
    channels = [32, 32] if norm_type == "group" else [8, 4]
    g = torch.Generator().manual_seed(seed + 8300)
    x = torch.randn(N, 4, 4, 4, 4, generator=g)
    model = SparseStructureDecoder(
        out_channels=2,
        latent_channels=4,
        num_res_blocks=1,
        channels=channels,
        num_res_blocks_middle=1,
        norm_type=norm_type,
    )
    _randomize(model, seed)
    with torch.no_grad():
        out = model(x)
    return {"x": x, "out": out, "sd": {k: v.detach() for k, v in model.state_dict().items()}}


def ref_slat_flow(seed, model_channels, num_heads, pe_mode, share_mod, qk_rms_norm, qk_rms_norm_cross, concat_channels=0):
    g = torch.Generator().manual_seed(seed + 8400)
    coords = gen_coords(seed, N, lo=3, hi=9)
    T = coords.shape[0]
    in_c = 6
    model = SLatFlowModel(
        resolution=16,
        in_channels=in_c,
        model_channels=model_channels,
        cond_channels=COND_C,
        out_channels=5,
        num_blocks=NUM_BLOCKS,
        num_heads=num_heads,
        mlp_ratio=MLP_RATIO,
        pe_mode=pe_mode,
        share_mod=share_mod,
        qk_rms_norm=qk_rms_norm,
        qk_rms_norm_cross=qk_rms_norm_cross,
    )
    _randomize(model, seed)
    feats = torch.randn(T, in_c - concat_channels, generator=g)
    t = torch.rand(N, generator=g) * 1000
    cond = torch.randn(N, COND_L, COND_C, generator=g)
    x = SparseTensor(feats=feats, coords=coords)
    ret = {"feats": feats, "coords": coords, "t": t, "cond": cond}
    with torch.no_grad():
        if concat_channels > 0:
            cc_feats = torch.randn(T, concat_channels, generator=g)
            ret["cc_feats"] = cc_feats
            out = model(x, t, cond, concat_cond=SparseTensor(feats=cc_feats, coords=coords))
        else:
            out = model(x, t, cond)
    ret["out"] = out.feats
    ret["sd"] = {k: v.detach() for k, v in model.state_dict().items()}
    return ret


UNET_CHANNELS = [16, 8]
UNET_NUM_BLOCKS = [1, 1]
UNET_LATENT = 4


def _unet_decoder(pred_subdiv):
    return SparseUnetVaeDecoder(
        out_channels=3,
        model_channels=UNET_CHANNELS,
        latent_channels=UNET_LATENT,
        num_blocks=UNET_NUM_BLOCKS,
        block_type=["SparseConvNeXtBlock3d"] * len(UNET_CHANNELS),
        up_block_type=["SparseResBlockC2S3d"] * (len(UNET_CHANNELS) - 1),
        block_args=[{}] * len(UNET_CHANNELS),
        pred_subdiv=pred_subdiv,
    )


def ref_unet_vae_decoders(seed):
    """pred_subdiv decoder + a guided decoder driven by its predicted subs,
    mirroring the pipeline's decode_shape_slat -> decode_tex_slat handoff."""
    g = torch.Generator().manual_seed(seed + 8500)
    coords = gen_coords(seed, N, lo=4, hi=10)
    T = coords.shape[0]
    feats = torch.randn(T, UNET_LATENT, generator=g)
    feats2 = torch.randn(T, UNET_LATENT, generator=g)

    # eval(): the torch forward has a training branch that returns 3 values
    dec = _randomize(_unet_decoder(True), seed).eval()
    dec2 = _randomize(_unet_decoder(False), seed + 1).eval()
    with torch.no_grad():
        out, subs = dec(SparseTensor(feats=feats, coords=coords), return_subs=True)
        out2 = dec2(SparseTensor(feats=feats2, coords=coords), guide_subs=subs)
        up_coords = dec.upsample(SparseTensor(feats=feats, coords=coords), upsample_times=1)
    return {
        "coords": coords, "feats": feats, "feats2": feats2,
        "out": out.feats, "out_coords": out.coords,
        "subs": [s.feats for s in subs],
        "out2": out2.feats, "out2_coords": out2.coords,
        "up_coords": up_coords,
        "sd": {k: v.detach() for k, v in dec.state_dict().items()},
        "sd2": {k: v.detach() for k, v in dec2.state_dict().items()},
    }


def ref_fdg_head(seed):
    g = torch.Generator().manual_seed(seed + 8600)
    feats = torch.randn(9, 7, generator=g) * 4
    margin = 0.5
    return {
        "feats": feats,
        "vertices": (1 + 2 * margin) * torch.sigmoid(feats[:, 0:3]) - margin,
        "intersected": (feats[:, 3:6] > 0).float(),
        "quad_lerp": torch.nn.functional.softplus(feats[:, 6:7]),
    }


def ref_flow_model(seed, model_channels, num_heads, pe_mode, share_mod, qk_rms_norm, qk_rms_norm_cross):
    x, t, cond = _inputs(seed)
    model = SparseStructureFlowModel(
        resolution=R,
        in_channels=IN_C,
        model_channels=model_channels,
        cond_channels=COND_C,
        out_channels=OUT_C,
        num_blocks=NUM_BLOCKS,
        num_heads=num_heads,
        mlp_ratio=MLP_RATIO,
        pe_mode=pe_mode,
        share_mod=share_mod,
        qk_rms_norm=qk_rms_norm,
        qk_rms_norm_cross=qk_rms_norm_cross,
    )
    _randomize(model, seed)
    with torch.no_grad():
        out = model(x, t, cond)
    return {
        "x": x, "t": t, "cond": cond, "out": out,
        "sd": {k: v.detach() for k, v in model.state_dict().items()},
    }
