# Real-checkpoint parity (WP9 del 3 steg 2): builds every pipeline model
# from the actual TRELLIS.2-4B / TRELLIS-image-large safetensors (bf16/fp16
# -> f32) through trellis2_mojo/checkpoints.mojo and compares full forwards
# against the torch originals loading the same files independently.
#
# Reads ~10 GB of checkpoints per run (each side loads its own copy), so
# this is its own task and NOT part of test-all:
#   pixi run test-real
#
# The 1024-res slat variants are skipped: identical architecture and keys
# to the 512 ones, only different weights — they exercise no new loading
# code. Tolerances are per-model: real weights + 30-block 1536-ch DiTs
# accumulate more drift than the synthetic wp8 cases (and the packed-GEMM
# linear is not bit-identical to torch's accumulation order), so max|diff|
# is printed on every run to keep the calibration honest.
#
# Run from repo root: pixi run mojo run -I . tests/parity/parity_real_ckpt_vs_torch.mojo

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_from_torch, tensor_to_torch, intmatrix_from_torch
from trellis2_mojo.checkpoints import (
    load_sparse_structure_flow,
    load_sparse_structure_decoder,
    load_slat_flow,
    load_unet_decoder,
)
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32


def check_close(name: String, m: Tensor[F32], expect: PythonObject, atol: Float64) raises:
    if Int(py=expect.numel()) != m.numel():
        raise Error(
            "size mismatch: " + name + " mojo " + String(m.numel())
            + " vs torch " + String(Int(py=expect.numel()))
        )
    var torch = Python.import_module("torch")
    var mt = tensor_to_torch(m)
    var d = Float64(py=(mt.reshape(expect.shape) - expect.to(torch.float32)).abs().max().item())
    print("  " + name + ": max|diff| " + String(d) + " (atol " + String(atol) + ")")
    if d > atol:
        raise Error("value mismatch: " + name)


def check_coords(name: String, m: IntMatrix, py: PythonObject) raises:
    if Int(py=py.shape[0]) != m.rows or Int(py=py.shape[1]) != m.cols:
        raise Error("coords shape mismatch: " + name)
    var expect = intmatrix_from_torch(py)
    for r in range(m.rows):
        for c in range(m.cols):
            if expect.at(r, c) != m.at(r, c):
                raise Error("coords mismatch: " + name + " @ row " + String(r))


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref_real")
    comptime seed = 0

    # -- SS-VAE decoder (smallest checkpoint; fully convolutional)
    print("ss_dec_conv3d_16l8_fp16 (sparse_structure_decoder):")
    var r0 = pyref.ref_ss_dec(seed)
    var ss_dec = load_sparse_structure_decoder()
    # occupancy logits are O(200) with real weights -> observed drift ~3e-4
    check_close("ss_dec", ss_dec.forward(tensor_from_torch(r0["x"])), r0["out"], 1e-3)

    # -- shape + tex UNet VAE decoders (the decode_shape -> decode_tex handoff)
    print("shape/tex_dec_next_dc_f16c32_fp16 (unet decoders):")
    var ru = pyref.ref_unet_decoders(seed)
    var shape_dec = load_unet_decoder("shape_slat_decoder")
    var x1 = SparseTensor[F32](tensor_from_torch(ru["feats"]), intmatrix_from_torch(ru["coords"]))
    var pair = shape_dec.forward(x1)
    check_close("shape_dec(out)", pair[0].vl.feats, ru["out"], 5e-4)
    check_coords("shape_dec(out)", pair[0].coords, ru["out_coords"])
    for i in range(len(pair[1])):
        check_close("shape_dec(subs " + String(i) + ")", pair[1][i].vl.feats, ru["subs_feats"][i], 5e-4)
        check_coords("shape_dec(subs " + String(i) + ")", pair[1][i].coords, ru["subs_coords"][i])

    var tex_dec = load_unet_decoder("tex_slat_decoder")
    var x2 = SparseTensor[F32](tensor_from_torch(ru["feats2"]), intmatrix_from_torch(ru["coords"]))
    # guide with the torch-side subs so tex parity is independent of any
    # borderline subdivision flips in the mojo shape run
    var guide = List[SparseTensor[F32]]()
    for i in range(len(pair[1])):
        guide.append(SparseTensor[F32](
            tensor_from_torch(ru["subs_feats"][i]), intmatrix_from_torch(ru["subs_coords"][i])
        ))
    var out2 = tex_dec.forward_guided(x2, guide)
    check_close("tex_dec(out)", out2.vl.feats, ru["out2"], 5e-4)
    check_coords("tex_dec(out)", out2.coords, ru["out2_coords"])

    # -- sparse DiTs, 512 variants (img2shape plain, imgshape2tex concat)
    print("slat_flow_img2shape_dit_1_3B_512_bf16:")
    var rs = pyref.ref_slat_flow(seed, "shape_slat_flow_model_512", 0)
    var shape_flow = load_slat_flow("shape_slat_flow_model_512")
    var xs = SparseTensor[F32](tensor_from_torch(rs["feats"]), intmatrix_from_torch(rs["coords"]))
    var out_s = shape_flow.forward(xs, tensor_from_torch(rs["t"]), tensor_from_torch(rs["cond"]))
    check_close("shape_flow", out_s.vl.feats, rs["out"], 2e-3)

    print("slat_flow_imgshape2tex_dit_1_3B_512_bf16:")
    var rt = pyref.ref_slat_flow(seed, "tex_slat_flow_model_512", 32)
    var tex_flow = load_slat_flow("tex_slat_flow_model_512")
    var xt = SparseTensor[F32](tensor_from_torch(rt["feats"]), intmatrix_from_torch(rt["coords"]))
    var cc = SparseTensor[F32](tensor_from_torch(rt["cc_feats"]), intmatrix_from_torch(rt["coords"]))
    var out_t = tex_flow.forward(xt, tensor_from_torch(rt["t"]), tensor_from_torch(rt["cond"]), cc)
    check_close("tex_flow", out_t.vl.feats, rt["out"], 2e-3)

    # -- dense DiT (heaviest forward: 4096 tokens x 30 blocks, both sides)
    print("ss_flow_img_dit_1_3B_64_bf16 (sparse_structure_flow):")
    var rf = pyref.ref_ss_flow(seed)
    var ss_flow = load_sparse_structure_flow()
    var out_f = ss_flow.forward(
        tensor_from_torch(rf["x"]), tensor_from_torch(rf["t"]), tensor_from_torch(rf["cond"])
    )
    check_close("ss_flow", out_f, rf["out"], 2e-3)

    print("real-checkpoint parity vs torch: ss_dec + shape/tex unet decoders + 3 DiTs passed")
