"""Reference side of the WP13 DINOv3 parity test.

Builds a SMALL random-weight DINOv3ViTModel straight from a transformers
config (no HF cache needed — safe for test-all) and runs it the way the
upstream DinoV3FeatureExtractor / cond_io does: embeddings -> rope ->
manual layer loop -> final F.layer_norm WITHOUT affine. The model's own
affine `norm` is deliberately not applied — that mirrors the extractor,
and it is what trellis2_mojo/models/dinov3.mojo ports.

_init_weights zeroes every bias and sets LayerNorm/LayerScale to identity,
which would leave those loader paths untested — so all parameters get an
extra seeded perturbation after init.

Real-checkpoint coverage (facebook/dinov3-vitl16, 24 layers) lives in
tests/parity/parity_cond_vs_torch.mojo (pixi run test-cond).
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
import torch.nn.functional as F
from transformers import DINOv3ViTConfig, DINOv3ViTModel

CFG = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,  # head_dim 16, rope inv_freq over 4 freqs
    intermediate_size=112,
    image_size=32,
    patch_size=8,
    num_register_tokens=3,
    rope_theta=100.0,
    hidden_act="gelu",
    use_gated_mlp=False,
    query_bias=True,
    key_bias=False,
    value_bias=True,
    proj_bias=True,
    mlp_bias=True,
    layerscale_value=1.0,
    attention_dropout=0.0,
    drop_path_rate=0.0,
)

_model = None


def model():
    global _model
    if _model is None:
        torch.manual_seed(0)
        _model = DINOv3ViTModel(DINOv3ViTConfig(**CFG)).eval()
        _model.config._attn_implementation = "eager"
        g = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            for p in _model.parameters():
                p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return _model


def state_dict():
    return {k: v.float() for k, v in model().state_dict().items()}


def pixel_case(seed, h, w):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(1, 3, int(h), int(w), generator=g)


@torch.no_grad()
def ref_features(seed, h, w):
    """Extractor-style forward on the seeded pixel case (see module doc)."""
    m = model()
    x = pixel_case(seed, h, w)
    hidden_states = m.embeddings(x, bool_masked_pos=None)
    position_embeddings = m.rope_embeddings(x)
    for layer_module in m.layer:
        hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
    return F.layer_norm(hidden_states, hidden_states.shape[-1:]).float()
