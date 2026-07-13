"""Reference side of the WP14 image-IO/preprocess parity test.

Exposes PIL/numpy ground truth for the pure-Mojo imaging stack:
  - Lanczos resize cases (PIL Image.resize LANCZOS) — the Mojo port
    mirrors Pillow's fixed-point 8bpc resampling, so comparisons are
    BIT-exact,
  - PAM/PPM raster round-trips (files written here, read by Mojo),
  - preprocess cases against cond_io.preprocess (itself asserted
    pixel-identical to the upstream alpha branch by torch_ref_cond.py),
    at 640 (no initial downscale) and 1500 (the >1024 downscale path),
  - normalized pixel tensors against cond_io.pixels.

u8 images cross the interop bridge as f32 torch tensors (values 0..255).
No HF cache or trellis2 import needed — safe for test-all.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from PIL import Image

from trellis2_mojo import cond_io

PAM_DIR = "/tmp"


def _write_pam(path, arr):
    depth = arr.shape[2]
    tupltype = "RGB_ALPHA" if depth == 4 else "RGB"
    header = (
        f"P7\nWIDTH {arr.shape[1]}\nHEIGHT {arr.shape[0]}\nDEPTH {depth}\n"
        f"MAXVAL 255\nTUPLTYPE {tupltype}\nENDHDR\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(arr.tobytes())
    return path


def _write_ppm(path, arr):
    with open(path, "wb") as f:
        f.write(f"P6\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode())
        f.write(arr.tobytes())
    return path


def _rand_img(seed, w, h, channels):
    g = np.random.default_rng(seed)
    return g.integers(0, 256, (h, w, channels), dtype=np.uint8)


def _t(arr):
    return torch.from_numpy(np.ascontiguousarray(arr)).float()


def resize_case(seed, w, h, channels, out_w, out_h):
    """-> (input u8 as f32 [h, w, c], PIL-expected u8 as f32 [out_h, out_w, c])."""
    arr = _rand_img(seed, int(w), int(h), int(channels))
    mode = "RGBA" if channels == 4 else "RGB"
    img = Image.fromarray(arr, mode)
    exp = np.array(img.resize((int(out_w), int(out_h)), Image.LANCZOS))
    return _t(arr), _t(exp)


def raster_case(seed, w, h, channels):
    """Writes a PAM (and a PPM when RGB) of a random image; -> (pam_path,
    ppm_path_or_empty, raw u8 as f32)."""
    arr = _rand_img(seed, int(w), int(h), int(channels))
    pam = _write_pam(f"{PAM_DIR}/trellis_mojo_wp14_{seed}.pam", arr)
    ppm = ""
    if channels == 3:
        ppm = _write_ppm(f"{PAM_DIR}/trellis_mojo_wp14_{seed}.ppm", arr)
    return pam, ppm, _t(arr)


def _make_test_image(seed=0, size=640):
    """Same construction as torch_ref_cond: gradient + blocks RGB, alpha =
    soft-edged disc with a fully transparent border."""
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    rgb = np.stack(
        [xx * 255, yy * 255, ((xx + yy) / 2 * 255)], axis=-1
    ) * 0.7 + g.uniform(0, 77, (size, size, 3))
    cy, cx, r = size * 0.55, size * 0.45, size * 0.3
    dist = np.sqrt((yy * size - cy) ** 2 + (xx * size - cx) ** 2)
    alpha = np.clip((r - dist) / (0.1 * r), 0, 1) * 255
    return np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.uint8)


def preprocess_case(size):
    """-> (pam_path, expected premultiplied RGB u8 as f32) via cond_io."""
    arr = _make_test_image(0, int(size))
    path = _write_pam(f"{PAM_DIR}/trellis_mojo_wp14_pre_{size}.pam", arr)
    pre = cond_io.preprocess(Image.fromarray(arr, "RGBA"))
    return path, _t(np.array(pre))


def pixels_case(size, resolution):
    """-> expected cond_io.pixels tensor [1, 3, R, R] f32."""
    arr = _make_test_image(0, int(size))
    pre = cond_io.preprocess(Image.fromarray(arr, "RGBA"))
    return cond_io.pixels(pre, int(resolution))


def opaque_pam():
    """All-opaque RGBA — must be rejected (ADR 0007)."""
    arr = _rand_img(9, 32, 32, 4)
    arr[:, :, 3] = 255
    return _write_pam(f"{PAM_DIR}/trellis_mojo_wp14_opaque.pam", arr)
