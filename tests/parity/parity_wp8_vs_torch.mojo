# WP8 parity vs the torch originals, weights loaded from state_dicts:
#   8.1 SparseStructureFlowModel (dense DiT) — APE, RoPE without phase
#       padding, RoPE with unit-phase padding, share_mod, qk_rms_norm.
#   8.2 SparseStructureDecoder (SS VAE) — layer and group norm, plus
#       standalone ResBlock3d with the 1x1 skip path both norm types.
#   8.3 SLatFlowModel (sparse DiT) — APE, RoPE+share_mod+qk_rms (real
#       checkpoint shape), concat_cond feature concat.
#   8.4 SparseUnetVaeDecoder — pred_subdiv decoder (feats+coords+subs),
#       guided decoder driven by those subs (the shape->tex handoff),
#       upsample() coords, and the FlexiDualGrid head transforms.
#
# Run from repo root: pixi run mojo run -I . tests/parity/parity_wp8_vs_torch.mojo

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_from_torch, intmatrix_from_torch
from trellis2_mojo.models.sparse_structure_flow import sparse_structure_flow_from
from trellis2_mojo.models.sparse_structure_vae import resblock3d_from, sparse_structure_decoder_from
from trellis2_mojo.models.structured_latent_flow import slat_flow_from
from trellis2_mojo.models.sc_vaes.sparse_unet_vae import sparse_unet_vae_decoder_from
from trellis2_mojo.models.sc_vaes.fdg_vae import fdg_head
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32

comptime RES = 4
comptime IN_C = 4
comptime OUT_C = 4
comptime COND_C = 12
comptime NUM_BLOCKS = 2


def check_tensor(name: String, m: Tensor[F32], py: PythonObject, atol: Float64 = 2e-4) raises:
    if Int(py=py.numel()) != m.numel():
        raise Error("size mismatch: " + name)
    var flat = py.flatten().tolist()
    for i in range(m.numel()):
        var d = Float64(py=flat[i]) - Float64(m.data[i])
        if d < 0:
            d = -d
        if d > atol:
            raise Error("value mismatch: " + name + " @ " + String(i) + " diff " + String(d))


def run_case(
    pyref: PythonObject,
    name: String,
    seed: Int,
    model_channels: Int,
    num_heads: Int,
    use_rope: Bool,
    share_mod: Bool,
    qk_rms_norm: Bool,
    qk_rms_norm_cross: Bool,
) raises:
    var pe_mode: String = "ape"
    if use_rope:
        pe_mode = "rope"
    var r = pyref.ref_flow_model(
        seed, model_channels, num_heads, pe_mode, share_mod, qk_rms_norm, qk_rms_norm_cross
    )
    var model = sparse_structure_flow_from(
        r["sd"], RES, IN_C, model_channels, COND_C, OUT_C, NUM_BLOCKS, num_heads,
        use_rope, share_mod, qk_rms_norm, qk_rms_norm_cross,
    )
    var out = model.forward(
        tensor_from_torch(r["x"]), tensor_from_torch(r["t"]), tensor_from_torch(r["cond"])
    )
    check_tensor(name, out, r["out"])


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


def run_slat_case(
    pyref: PythonObject,
    name: String,
    seed: Int,
    model_channels: Int,
    num_heads: Int,
    use_rope: Bool,
    share_mod: Bool,
    qk_rms_norm: Bool,
    qk_rms_norm_cross: Bool,
    concat_channels: Int,
) raises:
    var pe_mode: String = "ape"
    if use_rope:
        pe_mode = "rope"
    var r = pyref.ref_slat_flow(
        seed, model_channels, num_heads, pe_mode, share_mod, qk_rms_norm, qk_rms_norm_cross,
        concat_channels,
    )
    var model = slat_flow_from(
        r["sd"], 6, model_channels, COND_C, 5, NUM_BLOCKS, num_heads,
        use_rope, share_mod, qk_rms_norm, qk_rms_norm_cross,
    )
    var x = SparseTensor[F32](tensor_from_torch(r["feats"]), intmatrix_from_torch(r["coords"]))
    var out: SparseTensor[F32]
    if concat_channels > 0:
        var cc = SparseTensor[F32](tensor_from_torch(r["cc_feats"]), intmatrix_from_torch(r["coords"]))
        out = model.forward(x, tensor_from_torch(r["t"]), tensor_from_torch(r["cond"]), cc)
    else:
        out = model.forward(x, tensor_from_torch(r["t"]), tensor_from_torch(r["cond"]))
    check_tensor(name, out.vl.feats, r["out"])


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref_wp8")

    for seed in range(2):
        # APE, per-block adaLN, no qk norm (vanilla-style config)
        run_case(pyref, "flow(ape)", seed, 24, 2, False, False, False, False)
        # RoPE + share_mod + qk_rms both — mirrors the real ss_flow checkpoint
        # (head_dim 12: 3 axes x 2 freqs fill all 6 phase slots, no padding)
        run_case(pyref, "flow(rope,share,qk)", seed, 24, 2, True, True, True, True)
        # RoPE with unit-phase padding (head_dim 8: 3 slots + 1 pad slot)
        run_case(pyref, "flow(rope,pad)", seed, 24, 3, True, False, True, False)

        # -- WP8.2: ResBlock3d standalone (skip path), both norm types
        var rb1 = pyref.ref_resblock3d(seed, 5, 7, "layer")
        var b1 = resblock3d_from(rb1["sd"], "", 5, 7, False)
        check_tensor("resblock3d(layer,skip)", b1.forward(tensor_from_torch(rb1["x"])), rb1["out"])
        var rb2 = pyref.ref_resblock3d(seed, 32, 64, "group")
        var b2 = resblock3d_from(rb2["sd"], "", 32, 64, True)
        check_tensor("resblock3d(group,skip)", b2.forward(tensor_from_torch(rb2["x"])), rb2["out"])

        # -- WP8.2: SS VAE decoder, layer norm ([8, 4] stages)
        var rd = pyref.ref_ss_decoder(seed, "layer")
        var chans_l: List[Int] = [8, 4]
        var dec = sparse_structure_decoder_from(rd["sd"], 1, chans_l, 1, False)
        check_tensor("ss_decoder(layer)", dec.forward(tensor_from_torch(rd["x"])), rd["out"])

        # -- WP8.3: SLat flow — APE/vanilla, real-checkpoint shape, concat_cond
        run_slat_case(pyref, "slat(ape)", seed, 24, 2, False, False, False, False, 0)
        run_slat_case(pyref, "slat(rope,share,qk)", seed, 24, 2, True, True, True, True, 0)
        run_slat_case(pyref, "slat(concat)", seed, 24, 2, True, True, True, True, 2)

        # -- WP8.4: sparse UNet VAE — pred_subdiv decoder, then a guided
        # decoder driven by its predicted subs (shape->tex handoff)
        var ru = pyref.ref_unet_vae_decoders(seed)
        var chans_u: List[Int] = [16, 8]
        var nb_u: List[Int] = [1, 1]
        var d1 = sparse_unet_vae_decoder_from(ru["sd"], chans_u, nb_u, True)
        var x1 = SparseTensor[F32](tensor_from_torch(ru["feats"]), intmatrix_from_torch(ru["coords"]))
        var pair = d1.forward(x1)
        check_tensor("unet_dec(pred)", pair[0].vl.feats, ru["out"])
        check_coords("unet_dec(pred)", pair[0].coords, ru["out_coords"])
        check_tensor("unet_dec(subs)", pair[1][0].vl.feats, ru["subs"][0])
        check_coords("unet_dec(upsample)", d1.upsample_coords(x1, 1), ru["up_coords"])

        var d2 = sparse_unet_vae_decoder_from(ru["sd2"], chans_u, nb_u, False)
        var x2 = SparseTensor[F32](tensor_from_torch(ru["feats2"]), intmatrix_from_torch(ru["coords"]))
        var guide = List[SparseTensor[F32]]()
        guide.append(SparseTensor[F32](tensor_from_torch(ru["subs"][0]), intmatrix_from_torch(ru["coords"])))
        var out2 = d2.forward_guided(x2, guide)
        check_tensor("unet_dec(guided)", out2.vl.feats, ru["out2"])
        check_coords("unet_dec(guided)", out2.coords, ru["out2_coords"])

        # -- WP8.4: FlexiDualGrid head transforms
        var rf = pyref.ref_fdg_head(seed)
        var fcoords = IntMatrix(9, 4)
        for r in range(9):
            fcoords.set(r, 1, r)
        var hf = SparseTensor[F32](tensor_from_torch(rf["feats"]), fcoords^)
        var heads = fdg_head(hf)
        check_tensor("fdg_head(vertices)", heads[0].vl.feats, rf["vertices"])
        check_tensor("fdg_head(intersected)", heads[1].vl.feats, rf["intersected"])
        check_tensor("fdg_head(quad_lerp)", heads[2].vl.feats, rf["quad_lerp"])

    # group-norm decoder needs 32-multiple channels -> heavier weights; one seed
    var rdg = pyref.ref_ss_decoder(0, "group")
    var chans_g: List[Int] = [32, 32]
    var decg = sparse_structure_decoder_from(rdg["sd"], 1, chans_g, 1, True)
    check_tensor("ss_decoder(group)", decg.forward(tensor_from_torch(rdg["x"])), rdg["out"])

    print(
        "wp8 parity vs torch: flow (2 seeds x 3) + ss-vae decoder (layer/group)"
        + " + slat flow (2 seeds x 3) + unet-vae decoders + fdg head passed"
    )
