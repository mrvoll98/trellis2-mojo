# Conditioning parity (WP9 part 3 step 3; pure Mojo since WP13+WP14):
# ImageConditioner.get_cond — PAM decode + Mojo preprocess/Lanczos
# (bit-identical to PIL, see test-wp14) + Mojo ViT-L with real
# facebook/dinov3-vitl16 weights via the WP12 safetensors reader — vs the
# original DinoV3FeatureExtractor fed the SAME image through PIL/PNG, at
# a small resolution (128 -> 69 tokens) and the real pipeline resolution
# (512 -> 1029 tokens). Preprocess parity and the RGBA requirement are
# asserted inside torch_ref_cond.py at import.
#
# atol 5e-4: 24 f32 layers of accumulated op-order drift (GEMM path,
# sdpa) — same ladder as test-real's DiTs; outputs are O(1) after the
# final layer_norm. Small-config, tighter-tolerance coverage of the same
# module lives in test-wp13 (in test-all).
#
# Run from repo root: pixi run test-cond

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_to_torch
from trellis2_mojo.pipelines.conditioning import ImageConditioner, zeros_like_cond
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


def check_close(name: String, m: Tensor[F32], expected: PythonObject, atol: Float64) raises:
    var torch = Python.import_module("torch")
    var mt = tensor_to_torch(m)
    if Int(py=expected.numel()) != m.numel():
        raise Error("size mismatch: " + name)
    var diff = Float64(py=(mt.reshape(expected.shape) - expected).abs().max().item())
    print("  " + name + ": max|diff| " + String(diff) + " (atol " + String(atol) + ")")
    if diff > atol:
        raise Error("value mismatch: " + name)
    _ = Int(py=torch.numel(mt))


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    # import runs the preprocess-parity + RGBA-rejection asserts
    var pyref = Python.import_module("tests.parity.torch_ref_cond")
    var path = pyref._write_pam()  # pure-Mojo side reads PAM (WP14)

    var conditioner = ImageConditioner()

    var cond128 = conditioner.get_cond(String(py=path), 128)
    if cond128.shape[0] != 1 or cond128.shape[1] != 69 or cond128.shape[2] != 1024:
        raise Error("cond(128) shape mismatch")
    check_close("cond(128)", cond128, pyref.ref_cond(128), 5e-4)

    var neg = zeros_like_cond(cond128)
    if neg.numel() != cond128.numel():
        raise Error("neg_cond shape mismatch")
    for i in range(neg.numel()):
        if neg.data[i] != 0:
            raise Error("neg_cond not zero")

    var cond512 = conditioner.get_cond(String(py=path), 512)
    if cond512.shape[1] != 1029:
        raise Error("cond(512) token count mismatch")
    check_close("cond(512)", cond512, pyref.ref_cond(512), 5e-4)

    print("conditioning parity vs torch: preprocess (pixel-exact) + RGBA-krav + cond 128/512 passed")
