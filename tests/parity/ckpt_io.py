"""Real-checkpoint IO (WP9 del 3 steg 2): resolve the local HF cache and
load safetensors weights as float32 torch state_dicts for the Mojo loaders
(trellis2_mojo/checkpoints.mojo builds models from these via loaders.mojo).

Needs only torch + safetensors + stdlib and never touches the network —
raises if a checkpoint is missing from the cache (they were downloaded by
trellis-mac; see docs/08_HANDOVER.md for the asset map).
"""

import glob
import json
import os

import torch
from safetensors.torch import load_file

HF_HUB = os.path.join(
    os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")), "hub"
)

TRELLIS2_4B = "models--microsoft--TRELLIS.2-4B"
IMAGE_LARGE = "models--microsoft--TRELLIS-image-large"


def snapshot_dir(repo_dirname):
    snaps = sorted(glob.glob(os.path.join(HF_HUB, repo_dirname, "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError(f"no local HF snapshot for {repo_dirname} under {HF_HUB}")
    return snaps[-1]


def ckpt_base(name):
    """Resolve a model name to a checkpoint base path (no extension).

    Accepts the forms pipeline.json uses: "ckpts/<file>" (TRELLIS.2-4B
    relative), a bare "<file>", or "microsoft/TRELLIS-image-large/ckpts/
    <file>" (cross-repo reference).
    """
    if name.startswith("microsoft/TRELLIS-image-large/"):
        rel = name[len("microsoft/TRELLIS-image-large/"):]
        return os.path.join(snapshot_dir(IMAGE_LARGE), rel)
    if "/" not in name:
        name = "ckpts/" + name
    return os.path.join(snapshot_dir(TRELLIS2_4B), name)


def load_config(name):
    with open(ckpt_base(name) + ".json") as f:
        return json.load(f)


def load_state_dict_f32(name):
    """safetensors (bf16/fp16 on disk) -> dict of contiguous f32 CPU tensors.

    v1 of the Mojo port is float32-only, so the cast happens here, once,
    at load time (~2x the file size in RAM; load one model at a time).
    """
    sd = load_file(ckpt_base(name) + ".safetensors")
    return {k: v.to(torch.float32).contiguous() for k, v in sd.items()}


def pipeline_config():
    with open(os.path.join(snapshot_dir(TRELLIS2_4B), "pipeline.json")) as f:
        return json.load(f)


def model_path(model_key):
    """pipeline.json args.models[<key>] -> name for load_config/state_dict."""
    return pipeline_config()["args"]["models"][model_key]


# Small interpreters for config values that are awkward to test from Mojo.

def is_rope(args):
    return 1 if args.get("pe_mode") == "rope" else 0


def pred_subdiv(args):
    """SparseUnetVaeDecoder default is True; FlexiDualGridVaeDecoder (the
    shape decoder config) never sets it and relies on that default."""
    return 1 if args.get("pred_subdiv", True) else 0
