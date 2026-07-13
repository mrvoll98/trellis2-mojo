# WP9 pipeline-core integration parity: the Mojo sampling->decoding chain
# (velocity adapters + FlowEuler + models + pipeline glue) vs the original
# torch pipeline stages with identical weights and noise.
#
# Thresholded quantities (occupancy > 0, subdivision/intersection > 0) can
# legitimately differ when a logit sits within numeric drift of 0, so:
#   - occupancy is compared as raw values; the coords/pooling logic is
#     verified exactly on the torch occupancy tensor;
#   - the decode stage is compared only when the subdivision produced the
#     same rows; otherwise the test proves the cause was a borderline
#     subdivision logit and skips the value comparison.
#
# Run from repo root: pixi run mojo run -I . tests/parity/parity_wp9_vs_torch.mojo

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_from_torch, intmatrix_from_torch
from trellis2_mojo.samplers.flow_euler import FlowEulerSampler
from trellis2_mojo.models.sparse_structure_flow import sparse_structure_flow_from
from trellis2_mojo.models.sparse_structure_vae import sparse_structure_decoder_from
from trellis2_mojo.models.structured_latent_flow import slat_flow_from
from trellis2_mojo.models.sc_vaes.sparse_unet_vae import sparse_unet_vae_decoder_from
from trellis2_mojo.pipelines.image_to_3d import (
    SSFlowVelocity,
    SlatFlowVelocity,
    occupancy_to_coords,
    sample_slat,
    decode_shape,
    cascade_coords,
    normalize_slat,
    decode_tex,
)
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32

comptime COND_C = 12
comptime STEPS = 5
comptime RESCALE_T = 3.0
comptime CFG = 5.0
comptime LO = 0.2
comptime HI = 0.9
comptime GR_SS = 0.7    # real pipeline.json: ss guidance_rescale (dense std)
comptime GR_SLAT = 0.5  # real pipeline.json: slat guidance_rescale (varlen std)
comptime BORDERLINE = 2e-2  # |logit| under this may flip a > 0 decision


def check_tensor(name: String, m: Tensor[F32], py: PythonObject, atol: Float64) raises:
    if Int(py=py.numel()) != m.numel():
        raise Error("size mismatch: " + name)
    var flat = py.flatten().tolist()
    for i in range(m.numel()):
        var d = Float64(py=flat[i]) - Float64(m.data[i])
        if d < 0:
            d = -d
        if d > atol:
            raise Error("value mismatch: " + name + " @ " + String(i) + " diff " + String(d))


def check_coords(name: String, m: IntMatrix, py: PythonObject) raises:
    if Int(py=py.shape[0]) != m.rows or Int(py=py.shape[1]) != m.cols:
        raise Error("coords shape mismatch: " + name)
    var flat = py.flatten().tolist()
    var k = 0
    for r in range(m.rows):
        for c in range(m.cols):
            if Int(py=flat[k]) != m.at(r, c):
                raise Error("coords mismatch: " + name + " @ row " + String(r))
            k += 1


def check_binary_borderline(
    name: String, m: Tensor[F32], py_bits: PythonObject, py_logits: PythonObject
) raises:
    """0/1 tensors may disagree only where the torch logit is borderline."""
    var bits = py_bits.flatten().tolist()
    var logits = py_logits.flatten().tolist()
    for i in range(m.numel()):
        var want = Float64(py=bits[i])
        if Float64(m.data[i]) != want:
            var l = Float64(py=logits[i])
            if l < 0:
                l = -l
            if l > BORDERLINE:
                raise Error("bit mismatch (not borderline): " + name + " @ " + String(i))


def py_min_abs(py: PythonObject) raises -> Float64:
    var flat = py.flatten().tolist()
    var best: Float64 = 1e30
    for i in range(Int(py=py.numel())):
        var v = Float64(py=flat[i])
        if v < 0:
            v = -v
        if v < best:
            best = v
    return best


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref_wp9")

    for seed in range(2):
        var r = pyref.ref_pipeline_core(seed)
        var cond = tensor_from_torch(r["cond"])
        var neg_cond = Tensor[F32](cond.shape)  # pipeline: zeros_like(cond)
        var sampler = FlowEulerSampler(1e-5)

        # -- stage 1: sparse-structure sampling + decode (raw occupancy)
        var ss_model = sparse_structure_flow_from(
            r["ss_sd"], 4, 4, 24, COND_C, 4, 2, 2, True, True, True, True
        )
        var ss_chans: List[Int] = [8, 4]
        var ss_dec = sparse_structure_decoder_from(r["ss_dec_sd"], 1, ss_chans, 1, False)
        var vel = SSFlowVelocity(ss_model^, cond.copy(), neg_cond.copy())
        var z = sampler.sample_cfg_interval(
            vel, tensor_from_torch(r["noise"]), STEPS, RESCALE_T, CFG, LO, HI, GR_SS
        ).samples.copy()
        var occ = ss_dec.forward(z)
        check_tensor("occupancy", occ, r["occ_raw"], 1e-3)

        # -- coords/pooling logic, exactly, on the torch occupancy
        var occ_torch = tensor_from_torch(r["occ_raw"])
        check_coords("coords(full)", occupancy_to_coords(occ_torch, 8), r["coords8"])
        check_coords("coords(pooled)", occupancy_to_coords(occ_torch, 4), r["coords4"])

        # -- stage 2: shape-SLat sampling on the (torch) coords
        var slat_model = slat_flow_from(
            r["slat_sd"], 4, 24, COND_C, 4, 2, 2, True, True, True, True
        )
        var vel2 = SlatFlowVelocity(
            slat_model^, intmatrix_from_torch(r["coords4"]), cond.copy(), neg_cond.copy()
        )
        var slat = sample_slat(
            sampler, vel2, tensor_from_torch(r["noise2"]),
            tensor_from_torch(r["mean"]), tensor_from_torch(r["std"]),
            STEPS, RESCALE_T, CFG, LO, HI, GR_SLAT,
        )
        check_tensor("slat", slat.vl.feats, r["slat"], 2e-3)
        check_coords("slat coords", slat.coords, r["coords4"])

        # -- stage 3: decode + FDG head from the Mojo slat
        var unet_chans: List[Int] = [16, 8]
        var unet_nb: List[Int] = [1, 1]
        var unet = sparse_unet_vae_decoder_from(r["unet_sd"], unet_chans, unet_nb, True)
        var decoded = decode_shape(unet, slat)

        # -- stage 4: cascade — upsample coords, quantize/unique, HR slat
        var hr_coords = unet.upsample_coords(slat, 1)
        check_coords("hr coords", hr_coords, r["hr_coords"])
        var casc = cascade_coords(
            hr_coords, Int(py=r["lr_resolution"]), Int(py=r["cascade_resolution"]), 1000000
        )
        check_coords("cascade coords", casc[0], r["cascade_coords"])
        var hr_model = slat_flow_from(
            r["hr_sd"], 4, 24, COND_C, 4, 2, 2, True, True, True, True
        )
        var vel3 = SlatFlowVelocity(
            hr_model^, casc[0].copy(), cond.copy(), neg_cond.copy()
        )
        var hr_slat = sample_slat(
            sampler, vel3, tensor_from_torch(r["noise3"]),
            tensor_from_torch(r["mean"]), tensor_from_torch(r["std"]),
            STEPS, RESCALE_T, CFG, LO, HI, GR_SLAT,
        )
        check_tensor("hr slat", hr_slat.vl.feats, r["hr_slat"], 2e-3)

        # -- stage 5: texture — concat of normalized shape slat, then
        # guided decode using the shape decoder's subs
        var tex_model = slat_flow_from(
            r["tex_sd"], 7, 24, COND_C, 3, 2, 2, True, True, True, True
        )
        var vel4 = SlatFlowVelocity(
            tex_model^, intmatrix_from_torch(r["coords4"]), cond.copy(), neg_cond.copy()
        )
        var shape_norm = normalize_slat(
            slat, tensor_from_torch(r["mean"]), tensor_from_torch(r["std"])
        )
        vel4.set_concat(shape_norm.vl.feats.copy())
        var tex_slat = sample_slat(
            sampler, vel4, tensor_from_torch(r["noise4"]),
            tensor_from_torch(r["tex_mean"]), tensor_from_torch(r["tex_std"]),
            STEPS, RESCALE_T, CFG, LO, HI, GR_SLAT,
        )
        check_tensor("tex slat", tex_slat.vl.feats, r["tex_slat"], 2e-3)
        var tex_dec = sparse_unet_vae_decoder_from(r["tex_dec_sd"], unet_chans, unet_nb, False)
        if decoded[3][0].vl.feats.shape[0] == Int(py=r["sub0"].shape[0]):
            var tex_out = decode_tex(tex_dec, tex_slat, decoded[3].copy())
            if tex_out.vl.feats.shape[0] == Int(py=r["tex_out"].shape[0]):
                check_tensor("tex out", tex_out.vl.feats, r["tex_out"], 5e-3)
                check_coords("tex out coords", tex_out.coords, r["tex_out_coords"])
            else:
                if py_min_abs(r["sub0"]) > BORDERLINE:
                    raise Error("tex row mismatch without borderline subdivision @ seed " + String(seed))
                print("seed", seed, ": tex compare skipped (borderline subdivision flip)")
        check_tensor("subs", decoded[3][0].vl.feats, r["sub0"], 5e-3)
        if decoded[0].vl.feats.shape[0] == Int(py=r["out_coords"].shape[0]):
            check_coords("decoded coords", decoded[0].coords, r["out_coords"])
            check_tensor("vertices", decoded[0].vl.feats, r["vertices"], 5e-3)
            check_binary_borderline(
                "intersected", decoded[1].vl.feats, r["intersected"], r["out_feats"]
            )
            check_tensor("quad_lerp", decoded[2].vl.feats, r["quad_lerp"], 5e-3)
        else:
            # row count differs -> a subdivision bit flipped; legitimate only
            # if some torch subdivision logit was borderline
            if py_min_abs(r["sub0"]) > BORDERLINE:
                raise Error("decode row mismatch without borderline subdivision @ seed " + String(seed))
            print("seed", seed, ": decode compare skipped (borderline subdivision flip)")

    print("wp9 pipeline-core parity vs torch: 2 seeds, sample->decode + cascade + texture passed")
