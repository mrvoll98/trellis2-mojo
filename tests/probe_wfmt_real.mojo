# One-off probe (WP11 step 14): verify that REAL checkpoint weights select
# the expected device storage format — bf16 for the bf16 DiTs, f16 for the
# fp16 unet decoder. Reads ~3.3 GB from the HF cache; not a pixi task.
from trellis2_mojo.gpu.conv import GpuSparseConv
from trellis2_mojo.gpu.linear import (
    GpuContext, GpuLinear, WFMT_BF16, WFMT_F16, WFMT_F32,
)
from trellis2_mojo.io.hf_cache import ckpt_base
from trellis2_mojo.io.safetensors import load_safetensors_f32
from trellis2_mojo.io.state_dict import StateDict
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


def probe(gpu: Optional[GpuContext], sd: StateDict, key: String) raises -> Int:
    var w = sd.tensor(key)
    var b = Tensor[F32]([w.shape[0]])
    var gl = GpuLinear.try_build(gpu, w, b, False)
    if not gl:
        print("  ", key, w.shape[0], "x", w.shape[1], "-> DECLINED")
        return -1
    print("  ", key, w.shape[0], "x", w.shape[1], "-> wfmt", gl.value().wfmt)
    return gl.value().wfmt


def main() raises:
    var gpu: Optional[GpuContext] = GpuContext()
    print("ss_flow (bf16 ckpt):")
    var sd = StateDict(
        load_safetensors_f32(ckpt_base("ss_flow_img_dit_1_3B_64_bf16") + ".safetensors")
    )
    if probe(gpu, sd, "blocks.0.mlp.mlp.0.weight") != WFMT_BF16:
        raise Error("DiT mlp weight did not classify as bf16")
    if probe(gpu, sd, "blocks.0.self_attn.to_qkv.weight") != WFMT_BF16:
        raise Error("DiT qkv weight did not classify as bf16")
    print("shape_dec (fp16 ckpt):")
    var sd2 = StateDict(
        load_safetensors_f32(ckpt_base("shape_dec_next_dc_f16c32_fp16") + ".safetensors")
    )
    if probe(gpu, sd2, "blocks.0.0.mlp.0.weight") != WFMT_F16:
        raise Error("decoder ConvNeXt mlp weight did not classify as f16")
    var wc = sd2.tensor("blocks.0.0.conv.weight")
    var bc = Tensor[F32]([wc.shape[0]])
    var gc = GpuSparseConv.try_build(gpu, wc, bc, False)
    if not gc:
        raise Error("conv try_build declined the real decoder conv")
    print("   blocks.0.0.conv.weight [K,Ci,Co]", wc.shape[0], "-> wfmt", gc.value().wfmt)
    if gc.value().wfmt != WFMT_F16:
        raise Error("decoder conv weight did not classify as f16")
    print("real-ckpt wfmt probe passed")
