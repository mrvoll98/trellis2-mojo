# WP13 parity: trellis2_mojo/models/dinov3.mojo vs the transformers
# DINOv3ViTModel run extractor-style (manual layer loop + final non-affine
# layer_norm) on a small random-weight config — see torch_ref_wp13.py.
# Cache-independent and fast, so it lives in test-all.
#
# Cases: two square images and one non-square (3x5 patch grid) to catch
# y/x transposition in the 2D rope. atol 2e-5: two layers of f32 drift
# (WP8's ladder starts there); values are O(1) after the final layer_norm.
#
# Run from repo root: pixi run test-wp13

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_from_torch, tensor_to_torch
from trellis2_mojo.io.state_dict import StateDict
from trellis2_mojo.models.dinov3 import Dinov3ViT, dinov3_from
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


def run_case(model: Dinov3ViT, pyref: PythonObject, seed: Int, h: Int, w: Int, tokens: Int) raises:
    var x = tensor_from_torch(pyref.pixel_case(seed, h, w))
    var out = model.forward(x)
    if out.shape[0] != 1 or out.shape[1] != tokens or out.shape[2] != 64:
        raise Error("output shape mismatch")
    var name = "dinov3(seed " + String(seed) + ", " + String(h) + "x" + String(w) + ")"
    check_close(name, out, pyref.ref_features(seed, h, w), 2e-5)


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref_wp13")

    var sd = StateDict(pyref.state_dict())
    var model = dinov3_from(
        sd, num_layers=2, hidden=64, num_heads=4, patch_size=8,
        num_registers=3, rope_theta=100.0,
    )

    # 32x32 -> 4x4 patches, 1 + 3 + 16 = 20 tokens
    run_case(model, pyref, 0, 32, 32, 20)
    run_case(model, pyref, 1, 32, 32, 20)
    # non-square: 24x40 -> 3x5 patches, 1 + 3 + 15 = 19 tokens
    run_case(model, pyref, 2, 24, 40, 19)

    print("wp13 dinov3 parity vs torch: 2 square + 1 non-square case passed")
