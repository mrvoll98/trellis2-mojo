# Real-checkpoint loaders (WP9 del 3 steg 2, pure Mojo since WP12):
# resolve the local HF cache, parse the config JSONs and read the
# safetensors weights (bf16/fp16 -> f32) entirely in Mojo — no Python in
# this path anymore (trellis2_mojo/io/). Model keys are the pipeline.json
# names ("sparse_structure_flow_model", "shape_slat_flow_model_512",
# "shape_slat_decoder", ...).
#
# The weights are bf16/fp16 on disk and ~5.3 GB per DiT in f32: load one
# model at a time (see docs/08_HANDOVER.md). ckpt_io.py still exists, but
# only as the torch-side reference for the parity tests (test-io,
# test-real).

from trellis2_mojo.gpu.linear import GpuContext
from trellis2_mojo.io.hf_cache import model_path, ckpt_base, load_config_json, snapshot_dir, read_file_bytes
from trellis2_mojo.io.json import JsonDoc, parse_json
from trellis2_mojo.io.safetensors import load_safetensors_f32
from trellis2_mojo.io.state_dict import StateDict

from trellis2_mojo.models.dinov3 import Dinov3ViT, dinov3_from

from trellis2_mojo.models.sparse_structure_flow import (
    SparseStructureFlowModel,
    sparse_structure_flow_from,
)
from trellis2_mojo.models.sparse_structure_vae import (
    SparseStructureDecoder,
    sparse_structure_decoder_from,
)
from trellis2_mojo.models.structured_latent_flow import SLatFlowModel, slat_flow_from
from trellis2_mojo.models.sc_vaes.sparse_unet_vae import (
    SparseUnetVaeDecoder,
    sparse_unet_vae_decoder_from,
)


def _i(cfg: JsonDoc, a: Int, key: String) raises -> Int:
    return cfg.get_int(cfg.obj_get(a, key))


def _b(cfg: JsonDoc, a: Int, key: String) raises -> Bool:
    return cfg.get_bool(cfg.obj_get(a, key))


def _ints(cfg: JsonDoc, a: Int, key: String) raises -> List[Int]:
    var node = cfg.obj_get(a, key)
    var out = List[Int]()
    for i in range(cfg.arr_len(node)):
        out.append(cfg.get_int(cfg.arr_at(node, i)))
    return out^


def _is_rope(cfg: JsonDoc, a: Int) raises -> Bool:
    if not cfg.obj_has(a, "pe_mode"):
        return False
    return cfg.get_str(cfg.obj_get(a, "pe_mode")) == "rope"


def _pred_subdiv(cfg: JsonDoc, a: Int) raises -> Bool:
    """SparseUnetVaeDecoder default is True; FlexiDualGridVaeDecoder (the
    shape decoder config) never sets it and relies on that default."""
    if not cfg.obj_has(a, "pred_subdiv"):
        return True
    return cfg.get_bool(cfg.obj_get(a, "pred_subdiv"))


def _load(
    model_key: String, gpu: Optional[GpuContext]
) raises -> Tuple[JsonDoc, Int, StateDict]:
    """-> (config doc, args node, weights as StateDict). `gpu` rides on the
    StateDict so lin_from uploads device weights at build time (WP11)."""
    var name = model_path(model_key)
    var cfg = load_config_json(name)
    var a = cfg.obj_get(cfg.root, "args")
    var sd = StateDict(load_safetensors_f32(ckpt_base(name) + ".safetensors"))
    sd.gpu = gpu.copy()
    return (cfg^, a, sd^)


def load_sparse_structure_flow(
    gpu: Optional[GpuContext] = None,
) raises -> SparseStructureFlowModel:
    """ss_flow_img_dit_1_3B_64_bf16 -> dense DiT (rope + share_mod + qk_rms)."""
    var t = _load("sparse_structure_flow_model", gpu)
    var a = t[1]
    return sparse_structure_flow_from(
        t[2],
        _i(t[0], a, "resolution"), _i(t[0], a, "in_channels"), _i(t[0], a, "model_channels"),
        _i(t[0], a, "cond_channels"), _i(t[0], a, "out_channels"), _i(t[0], a, "num_blocks"),
        _i(t[0], a, "num_heads"),
        use_rope=_is_rope(t[0], a),
        share_mod=_b(t[0], a, "share_mod"),
        qk_rms_norm=_b(t[0], a, "qk_rms_norm"),
        qk_rms_norm_cross=_b(t[0], a, "qk_rms_norm_cross"),
    )


def load_sparse_structure_decoder() raises -> SparseStructureDecoder:
    """ss_dec_conv3d_16l8_fp16 (TRELLIS-image-large) -> SS-VAE decoder.

    The config carries no norm_type, so the out_layer uses the upstream
    default "layer" (res blocks always use "layer" regardless — finding 8).
    """
    var t = _load("sparse_structure_decoder", None)
    var a = t[1]
    return sparse_structure_decoder_from(
        t[2],
        _i(t[0], a, "num_res_blocks"), _ints(t[0], a, "channels"),
        _i(t[0], a, "num_res_blocks_middle"), norm_group=False,
    )


def load_slat_flow(
    model_key: String, gpu: Optional[GpuContext] = None
) raises -> SLatFlowModel:
    """slat_flow_{img2shape,imgshape2tex}_dit_1_3B_{512,1024}_bf16 -> sparse
    DiT; the tex variants take the shape latent via concat_cond
    (in_channels 64 = 32 slat + 32 concat)."""
    var t = _load(model_key, gpu)
    var a = t[1]
    return slat_flow_from(
        t[2],
        _i(t[0], a, "in_channels"), _i(t[0], a, "model_channels"), _i(t[0], a, "cond_channels"),
        _i(t[0], a, "out_channels"), _i(t[0], a, "num_blocks"), _i(t[0], a, "num_heads"),
        use_rope=_is_rope(t[0], a),
        share_mod=_b(t[0], a, "share_mod"),
        qk_rms_norm=_b(t[0], a, "qk_rms_norm"),
        qk_rms_norm_cross=_b(t[0], a, "qk_rms_norm_cross"),
    )


comptime DINOV3_REPO = "models--facebook--dinov3-vitl16-pretrain-lvd1689m"


def load_dinov3(gpu: Optional[GpuContext] = None) raises -> Dinov3ViT:
    """facebook/dinov3-vitl16-pretrain-lvd1689m -> pure-Mojo image encoder
    (WP13). Lives in its own HF repo (unlike the pipeline models, which
    pipeline.json names); config + f32 safetensors sit in the snapshot
    root. ~1.2 GB resident."""
    var snap = snapshot_dir(DINOV3_REPO)
    var cfg = parse_json(read_file_bytes(snap + "/config.json"))
    var a = cfg.root
    var sd = StateDict(load_safetensors_f32(snap + "/model.safetensors"))
    sd.gpu = gpu.copy()
    return dinov3_from(
        sd,
        _i(cfg, a, "num_hidden_layers"), _i(cfg, a, "hidden_size"),
        _i(cfg, a, "num_attention_heads"), _i(cfg, a, "patch_size"),
        _i(cfg, a, "num_register_tokens"),
        rope_theta=cfg.get_float(cfg.obj_get(a, "rope_theta")),
        eps=cfg.get_float(cfg.obj_get(a, "layer_norm_eps")),
    )


def load_unet_decoder(
    model_key: String, gpu: Optional[GpuContext] = None
) raises -> SparseUnetVaeDecoder:
    """{shape,tex}_dec_next_dc_f16c32_fp16 -> sparse UNet VAE decoder.

    The shape config is a FlexiDualGridVaeDecoder (out_channels=7 baked in,
    pred_subdiv defaulting True); out/latent channels are implied by the
    weights, so only pred_subdiv needs interpreting here. The FDG head
    itself is weight-free (fdg_vae.mojo)."""
    var t = _load(model_key, gpu)
    var a = t[1]
    return sparse_unet_vae_decoder_from(
        t[2],
        _ints(t[0], a, "model_channels"), _ints(t[0], a, "num_blocks"),
        pred_subdiv=_pred_subdiv(t[0], a),
    )
