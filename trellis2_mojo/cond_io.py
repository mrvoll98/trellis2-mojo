"""PIL/torch-side reference for the conditioning parity tests.

NOT in the runner path anymore: since WP13 the DINOv3 transformer runs in
pure Mojo (models/dinov3.mojo) and since WP14 so does the whole image
path (io/image.mojo PAM/PPM decode + imaging/ preprocess/Lanczos/
normalize). This module mirrors the alpha branch of
`Trellis2ImageTo3DPipeline.preprocess_image` and the extractor's pixel
prep exactly (asserted against the originals by torch_ref_cond.py) and
serves as ground truth for test-wp14/test-cond — the same role ckpt_io.py
plays for test-io/test-real.

Per ADR 0007 the rembg/BiRefNet path is intentionally NOT supported: input
images must be RGBA with a real alpha channel (object already cut out).
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
from PIL import Image

DINOV3_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(image):
    """Alpha branch of the upstream preprocess_image (rembg branch removed).

    Takes a path or PIL image; requires RGBA with a non-trivial alpha
    channel (ADR 0007). Returns the premultiplied RGB PIL image.
    """
    if isinstance(image, str):
        image = Image.open(image)
    if image.mode != "RGBA" or np.all(np.array(image)[:, :, 3] == 255):
        raise ValueError(
            "input must be RGBA with a real alpha channel (background removed); "
            "the rembg path is not supported per ADR 0007"
        )
    max_size = max(image.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS
        )
    output_np = np.array(image)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    output = image.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    return Image.fromarray((output * 255).astype(np.uint8))


def pixels(image, resolution):
    """Preprocessed image (path or PIL) -> normalized pixels [1, 3, R, R] f32.

    Mirrors DinoV3FeatureExtractor's input prep exactly: LANCZOS resize to
    resolution^2, /255, ImageNet normalize. The transformer itself runs in
    Mojo (models/dinov3.mojo) — L = 1 cls + 4 registers + (resolution/16)^2
    patches (1029 at 512, 4101 at 1024).
    """
    if isinstance(image, str):
        image = Image.open(image)
    image = image.resize((resolution, resolution), Image.LANCZOS)
    x = np.array(image.convert("RGB")).astype(np.float32) / 255
    t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (t - mean) / std
