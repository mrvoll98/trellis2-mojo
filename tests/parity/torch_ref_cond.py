"""Reference side of the conditioning parity test (WP9 part 3 step 3).

Builds a deterministic synthetic RGBA image, then checks trellis2_mojo/
cond_io.py against the ORIGINALS on import-time asserts:
  - preprocess() must be pixel-identical to the alpha branch of
    Trellis2ImageTo3DPipeline.preprocess_image (called with a dummy self —
    the alpha branch never touches self),
  - RGB / all-opaque input must be rejected (ADR 0007: no rembg path),
and exposes DinoV3FeatureExtractor features on the preprocessed image for
the Mojo side to compare against ImageConditioner.get_cond.

Both extractors run the same transformers weights; what is under test is
the wrapper logic (resize/normalize/layer loop/final layer_norm) and the
Mojo interface. HF_HUB_OFFLINE is set by cond_io — everything loads from
the local cache.
"""

import os
import sys

os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
os.environ.setdefault("ATTN_BACKEND", "naive")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from trellis2_mojo import cond_io  # noqa: E402  (sets HF offline env guards)
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor  # noqa: E402
from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline  # noqa: E402

IMAGE_PATH = "/tmp/trellis_mojo_test_cond.png"
PAM_PATH = "/tmp/trellis_mojo_test_cond.pam"


def _make_test_image(seed=0, size=640):
    """Gradient + blocks RGB, alpha = centered disc with soft edge and a
    fully transparent border — exercises bbox crop and premultiply."""
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    rgb = np.stack(
        [xx * 255, yy * 255, ((xx + yy) / 2 * 255)], axis=-1
    ) * 0.7 + g.uniform(0, 77, (size, size, 3))
    cy, cx, r = size * 0.55, size * 0.45, size * 0.3
    dist = np.sqrt((yy * size - cy) ** 2 + (xx * size - cx) ** 2)
    alpha = np.clip((r - dist) / (0.1 * r), 0, 1) * 255
    img = np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.uint8)
    return Image.fromarray(img, mode="RGBA")


def _write_image():
    _make_test_image().save(IMAGE_PATH)
    return IMAGE_PATH


def _write_pam():
    """Same test image as PAM P7 — the pure-Mojo runner input (WP14)."""
    arr = np.array(_make_test_image())
    header = (
        f"P7\nWIDTH {arr.shape[1]}\nHEIGHT {arr.shape[0]}\nDEPTH 4\n"
        "MAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
    )
    with open(PAM_PATH, "wb") as f:
        f.write(header.encode())
        f.write(arr.tobytes())
    return PAM_PATH


class _DummyPipe:
    low_vram = False
    rembg_model = None


# --- import-time asserts: preprocess parity + RGBA requirement ---
_img = _make_test_image()
_ours = np.array(cond_io.preprocess(_img))
_orig = np.array(Trellis2ImageTo3DPipeline.preprocess_image(_DummyPipe(), _img))
assert _ours.shape == _orig.shape, f"preprocess shape {_ours.shape} != {_orig.shape}"
assert np.array_equal(_ours, _orig), "preprocess not pixel-identical to the original"

for _bad in (_make_test_image().convert("RGB"), Image.new("RGBA", (64, 64), (10, 20, 30, 255))):
    try:
        cond_io.preprocess(_bad)
        raise AssertionError("preprocess accepted input without a real alpha channel")
    except ValueError:
        pass

_extractor = None


def ref_cond(resolution):
    """Original extractor's features on the preprocessed test image."""
    global _extractor
    if _extractor is None:
        _extractor = DinoV3FeatureExtractor(cond_io.DINOV3_NAME)
    _extractor.image_size = resolution
    image = cond_io.preprocess(_make_test_image())
    with torch.no_grad():
        return _extractor([image]).float()
