"""Reference side of the real-checkpoint parity test (WP9 del 3 steg 2).

Loads the actual TRELLIS.2-4B / TRELLIS-image-large checkpoints from the
local HF cache (bf16/fp16 -> f32) into the ORIGINAL torch models and runs
seeded forwards. The Mojo side loads the same files independently through
trellis2_mojo/checkpoints.mojo, so the key mapping and weight fidelity of
the whole loading path are verified end to end, on top of forward parity
with real weights.

Importing torch_ref_wp5 patches the original's buggy CPU sparse-sdpa
fallback (finding 1). The batches here are B=1, where the unpatched
fallback happens to be harmless, but the patch keeps us on the verified
path either way.

Each ref_* call instantiates one model, runs it and frees it before
returning — peak is one 1.3B model in f32 (~5.3 GB) plus activations.
The shape decoder is built as a plain SparseUnetVaeDecoder with the
FlexiDualGridVaeDecoder defaults (out_channels=7, pred_subdiv=True):
importing the FDG class would pull in the o_voxel mesh extractor, which
is step 4's problem, and the subclass adds no weights.
"""

import os
import sys

os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
os.environ.setdefault("ATTN_BACKEND", "naive")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from tests.parity.torch_ref_wp5 import gen_coords, correct_sdpa  # noqa: E402,F401  (patches sparse attn)
from trellis2_mojo import ckpt_io  # noqa: E402

from trellis2.modules.sparse.basic import SparseTensor  # noqa: E402
from trellis2.models.sparse_structure_flow import SparseStructureFlowModel  # noqa: E402
from trellis2.models.sparse_structure_vae import SparseStructureDecoder  # noqa: E402
from trellis2.models.structured_latent_flow import SLatFlowModel  # noqa: E402
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder  # noqa: E402

COND_L = 7  # arbitrary short cond sequence; parity only needs matching inputs


def _args(name, drop=(), **overrides):
    args = dict(ckpt_io.load_config(name)["args"])
    for k in ("dtype", "use_fp16", "initialization") + tuple(drop):
        args.pop(k, None)
    args.update(overrides)
    return args


def _load(cls, name, drop=(), **overrides):
    model = cls(**_args(name, drop, **overrides))
    # strict=False mirrors the original from_pretrained (models/__init__.py)
    model.load_state_dict(ckpt_io.load_state_dict_f32(name), strict=False)
    return model.eval()


def ref_ss_dec(seed):
    """Fully convolutional -> a small latent keeps the runtime sane."""
    name = ckpt_io.model_path("sparse_structure_decoder")
    a = _args(name)
    g = torch.Generator().manual_seed(seed + 9000)
    x = torch.randn(1, a["latent_channels"], 4, 4, 4, generator=g)
    model = _load(SparseStructureDecoder, name)
    with torch.no_grad():
        out = model(x)
    del model
    return {"x": x, "out": out}


def ref_ss_flow(seed):
    name = ckpt_io.model_path("sparse_structure_flow_model")
    a = _args(name)
    g = torch.Generator().manual_seed(seed + 9100)
    r = a["resolution"]
    x = torch.randn(1, a["in_channels"], r, r, r, generator=g)
    t = torch.rand(1, generator=g) * 1000
    cond = torch.randn(1, COND_L, a["cond_channels"], generator=g)
    model = _load(SparseStructureFlowModel, name)
    with torch.no_grad():
        out = model(x, t, cond)
    del model
    return {"x": x, "t": t, "cond": cond, "out": out}


def ref_slat_flow(seed, model_key, concat_channels):
    """concat_channels=32 for the tex variants (in_ch 64 = 32 slat + 32
    concat, the pipeline's shape-latent conditioning), 0 for img2shape."""
    name = ckpt_io.model_path(model_key)
    a = _args(name)
    g = torch.Generator().manual_seed(seed + 9200)
    coords = gen_coords(seed, 1, grid=a["resolution"], lo=180, hi=260)
    T = coords.shape[0]
    feats = torch.randn(T, a["in_channels"] - concat_channels, generator=g)
    t = torch.rand(1, generator=g) * 1000
    cond = torch.randn(1, COND_L, a["cond_channels"], generator=g)
    model = _load(SLatFlowModel, name)
    ret = {"feats": feats, "coords": coords, "t": t, "cond": cond}
    with torch.no_grad():
        if concat_channels > 0:
            cc = torch.randn(T, concat_channels, generator=g)
            ret["cc_feats"] = cc
            out = model(SparseTensor(feats=feats, coords=coords), t, cond,
                        concat_cond=SparseTensor(feats=cc, coords=coords))
        else:
            out = model(SparseTensor(feats=feats, coords=coords), t, cond)
    del model
    ret["out"] = out.feats
    return ret


def ref_unet_decoders(seed):
    """Shape decoder (pred_subdiv) then the tex decoder guided by its subs —
    the pipeline's decode_shape -> decode_tex handoff, with real weights.

    Every upsampling level must keep tokens or the graphs collapse; with
    random latents and real weights that is seed-dependent, so fail loudly
    here rather than mysteriously in Mojo."""
    shape_name = ckpt_io.model_path("shape_slat_decoder")
    tex_name = ckpt_io.model_path("tex_slat_decoder")
    sa = _args(shape_name, drop=("resolution",))
    ta = _args(tex_name)
    g = torch.Generator().manual_seed(seed + 9300)
    coords = gen_coords(seed, 1, grid=16, lo=12, hi=20)
    T = coords.shape[0]
    feats = torch.randn(T, sa["latent_channels"], generator=g)
    feats2 = torch.randn(T, ta["latent_channels"], generator=g)

    dec = _load(SparseUnetVaeDecoder, shape_name, drop=("resolution",),
                out_channels=7, pred_subdiv=True)
    with torch.no_grad():
        out, subs = dec(SparseTensor(feats=feats, coords=coords), return_subs=True)
    del dec
    for i, s in enumerate(subs):
        assert s.feats.shape[0] > 0, f"subdivision level {i} kept no tokens (seed {seed})"
    assert out.feats.shape[0] > 0, f"decoder output empty (seed {seed})"

    dec2 = _load(SparseUnetVaeDecoder, tex_name)
    with torch.no_grad():
        out2 = dec2(SparseTensor(feats=feats2, coords=coords), guide_subs=subs)
    del dec2
    return {
        "coords": coords, "feats": feats, "feats2": feats2,
        "out": out.feats, "out_coords": out.coords,
        "subs_feats": [s.feats for s in subs],
        "subs_coords": [s.coords for s in subs],
        "out2": out2.feats, "out2_coords": out2.coords,
    }
