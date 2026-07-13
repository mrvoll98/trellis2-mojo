# End-to-end image->3D runner (WP9 del 3 steg 5) — Mojo host.
#
# The whole pipeline runs in Mojo: PAM decode + preprocess (io/image,
# imaging/, WP14 — bit-identical to the PIL originals), safetensors/config
# loading (trellis2_mojo/io/, WP12), DINOv3 conditioning
# (models/dinov3.mojo, WP13), FlowEuler sampling (ss -> shape slat -> tex
# slat), SS-VAE + UNet-VAE decoding, FDG head and pure-Mojo mesh
# extraction + OBJ export. The only Python left (per ADR 0007):
# torch.randn, so noise is drawn from the same seeded stream as the
# original pipeline / trellis-mac.
#
# Mirrors Trellis2ImageTo3DPipeline.run(pipeline_type='512'): sampler
# params and slat normalizations come from pipeline.json, models load from
# the local HF cache one at a time (~5.3 GB per DiT in f32; each stage is
# its own function so the model frees before the next loads).
#
# Noise-stream note: upstream seeds BEFORE get_cond, but conditioning
# consumes no RNG at inference — only model CONSTRUCTION draws from the
# global generator, and upstream constructs the extractor at
# from_pretrained time, pre-seed. We therefore compute cond first and seed
# after, which reproduces the upstream draw order exactly:
# randn(1, C, 16, 16, 16) -> randn(N, 32) -> randn(N, 32).
#
# Input must be a PAM P7 file, RGBA with a real alpha channel (ADR 0007 —
# no rembg). PNG -> PAM is a one-liner, see README_MOJO.md.
# Run from the repo root:
#   pixi run e2e -- <image.pam> [--seed N] [--steps N] [--out prefix] [--no-tex]

from std.python import Python, PythonObject
from std.sys import argv
from std.time import perf_counter_ns

from trellis2_mojo.gpu.linear import GpuContext, gpu_context_from_env
from trellis2_mojo.interop import tensor_from_torch, tensor_to_torch
from trellis2_mojo.checkpoints import (
    load_sparse_structure_flow,
    load_sparse_structure_decoder,
    load_slat_flow,
    load_unet_decoder,
)
from trellis2_mojo.io.hf_cache import pipeline_config_json
from trellis2_mojo.io.json import JsonDoc
from trellis2_mojo.pipelines.conditioning import ImageConditioner, zeros_like_cond
from trellis2_mojo.pipelines.image_to_3d import (
    SSFlowVelocity,
    SlatFlowVelocity,
    cascade_coords,
    sample_sparse_structure,
    sample_slat,
    decode_shape,
    normalize_slat,
    decode_tex,
)
from trellis2_mojo.samplers.flow_euler import FlowEulerSampler
from trellis2_mojo.io.glb import to_glb_axes, write_glb
from trellis2_mojo.meshing.fdg_mesh import flexible_dual_grid_to_mesh, write_obj
from trellis2_mojo.meshing.postprocess import (
    fill_small_holes,
    remove_small_connected_components,
    repair_non_manifold_edges,
    sew_boundary_seams,
    unify_face_orientations,
)
from trellis2_mojo.meshing.remesh import remesh_narrow_band_dc
from trellis2_mojo.meshing.vertex_attrs import (
    grid_sample_trilinear,
    vertex_normals,
)
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix
from trellis2_mojo.sparse.basic import SparseTensor

comptime F32 = DType.float32


struct SamplerParams(Copyable, Movable):
    """One stage's block from pipeline.json (steps overridable from CLI)."""

    var steps: Int
    var rescale_t: Float64
    var strength: Float64
    var interval_lo: Float64
    var interval_hi: Float64
    var rescale: Float64
    var sigma_min: Float64

    def __init__(out self, doc: JsonDoc, args_node: Int, key: String, steps_override: Int) raises:
        var stage = doc.obj_get(args_node, key)
        var p = doc.obj_get(stage, "params")
        var interval = doc.obj_get(p, "guidance_interval")
        self.steps = steps_override if steps_override > 0 else doc.get_int(doc.obj_get(p, "steps"))
        self.rescale_t = doc.get_float(doc.obj_get(p, "rescale_t"))
        self.strength = doc.get_float(doc.obj_get(p, "guidance_strength"))
        self.interval_lo = doc.get_float(doc.arr_at(interval, 0))
        self.interval_hi = doc.get_float(doc.arr_at(interval, 1))
        self.rescale = doc.get_float(doc.obj_get(p, "guidance_rescale"))
        self.sigma_min = doc.get_float(doc.obj_get(doc.obj_get(stage, "args"), "sigma_min"))


def _norm_tensor(doc: JsonDoc, args_node: Int, key: String, field: String) raises -> Tensor[F32]:
    """pipeline.json {shape,tex}_slat_normalization mean/std -> Tensor [C]."""
    var node = doc.obj_get(doc.obj_get(args_node, key), field)
    var n = doc.arr_len(node)
    var t = Tensor[F32]([n])
    for i in range(n):
        t.data[i] = Float32(doc.get_float(doc.arr_at(node, i)))
    return t^


def parse_int(s: String) raises -> Int:
    return Int(py=Python.import_module("builtins").int(s))


def parse_float(s: String) raises -> Float64:
    return Float64(py=Python.import_module("builtins").float(s))


def elapsed(t0: UInt) -> String:
    var s = Float64(perf_counter_ns() - t0) / 1e9
    return String(Int(s)) + "s"


def run_ss_stage(
    torch: PythonObject, cond: Tensor[F32], neg: Tensor[F32],
    p: SamplerParams, ss_res: Int, gpu: Optional[GpuContext],
) raises -> IntMatrix:
    """Stage 1: dense DiT sampling + SS-VAE decode -> active voxel coords."""
    var t0 = perf_counter_ns()
    var model = load_sparse_structure_flow(gpu)
    print("  [ss] dit load:", elapsed(t0))
    var noise = tensor_from_torch(
        torch.randn(1, model.in_channels, model.resolution, model.resolution, model.resolution)
    )
    t0 = perf_counter_ns()
    var decoder = load_sparse_structure_decoder()
    print("  [ss] decoder load:", elapsed(t0))
    var sampler = FlowEulerSampler(p.sigma_min)
    var vel = SSFlowVelocity(model^, cond.copy(), neg.copy())
    t0 = perf_counter_ns()
    var coords = sample_sparse_structure(
        sampler, vel, decoder, noise, ss_res,
        p.steps, p.rescale_t, p.strength, p.interval_lo, p.interval_hi, p.rescale,
    )
    print("  [ss] sample+decode:", elapsed(t0))
    return coords^


def run_shape_slat_stage(
    torch: PythonObject, model_key: String, cond: Tensor[F32], neg: Tensor[F32],
    coords: IntMatrix, mean: Tensor[F32], std: Tensor[F32], p: SamplerParams,
    gpu: Optional[GpuContext],
) raises -> SparseTensor[F32]:
    """Stage 2: sparse DiT on the fixed coords -> de-normalized shape slat.
    model_key selects the 512 or 1024 DiT (the cascade samples both)."""
    var t0 = perf_counter_ns()
    var model = load_slat_flow(model_key, gpu)
    print("  [slat] dit load:", elapsed(t0))
    var noise = tensor_from_torch(torch.randn(coords.rows, model.in_channels))
    var sampler = FlowEulerSampler(p.sigma_min)
    var vel = SlatFlowVelocity(model^, coords.copy(), cond.copy(), neg.copy())
    t0 = perf_counter_ns()
    var slat = sample_slat(
        sampler, vel, noise, mean, std,
        p.steps, p.rescale_t, p.strength, p.interval_lo, p.interval_hi, p.rescale,
    )
    print("  [slat] sample:", elapsed(t0))
    return slat^


def run_cascade_stage(
    lr_slat: SparseTensor[F32], gpu: Optional[GpuContext]
) raises -> IntMatrix:
    """1024-cascade (upstream sample_shape_slat_cascade's middle part):
    the shape decoder predicts subdivisions 4 levels up from the LR slat
    (32^3 coords -> 512-res coords), which quantize to 64^3 and dedupe
    into the HR token coords. The decoder is loaded just for the
    upsample and freed again (one model in memory at a time); for the
    1024 target the token-budget loop is a no-op (it always breaks at
    hr_resolution == 1024)."""
    var t0 = perf_counter_ns()
    var decoder = load_unet_decoder("shape_slat_decoder", gpu)
    var hr = decoder.upsample_coords(lr_slat, 4)
    var pair = cascade_coords(hr, 512, 1024, 49152)
    print("  [slat] cascade upsample -> 64^3:", pair[0].rows, "tokens in", elapsed(t0))
    return pair[0].copy()


def run_tex_slat_stage(
    torch: PythonObject, model_key: String, cond: Tensor[F32], neg: Tensor[F32],
    shape_slat: SparseTensor[F32],
    shape_mean: Tensor[F32], shape_std: Tensor[F32],
    tex_mean: Tensor[F32], tex_std: Tensor[F32], p: SamplerParams,
    gpu: Optional[GpuContext],
) raises -> SparseTensor[F32]:
    """Stage 3: shape slat re-normalized with the SHAPE stats rides as
    concat_cond; the sampled texture slat de-normalizes with the TEX stats."""
    var model = load_slat_flow(model_key, gpu)
    var noise = tensor_from_torch(
        torch.randn(shape_slat.coords.rows, model.in_channels - shape_slat.vl.feats.shape[1])
    )
    var sampler = FlowEulerSampler(p.sigma_min)
    var vel = SlatFlowVelocity(
        model^, shape_slat.coords.copy(), cond.copy(), neg.copy()
    )
    var shape_norm = normalize_slat(shape_slat, shape_mean, shape_std)
    vel.set_concat(shape_norm.vl.feats.copy())
    return sample_slat(
        sampler, vel, noise, tex_mean, tex_std,
        p.steps, p.rescale_t, p.strength, p.interval_lo, p.interval_hi, p.rescale,
    )


def save_tex_voxels(
    torch: PythonObject, path: String, tex: SparseTensor[F32], resolution: Int
) raises:
    """Texture voxels -> npz (coords int32 [T,3], attrs f32 [T,6] in the
    upstream pbr layout base_color/metallic/roughness/alpha, plus origin +
    voxel_size) — the payload MeshWithVoxel carries for texture baking."""
    var np = Python.import_module("numpy")
    var t = tex.coords.rows
    var cf = Tensor[F32]([t, 3])
    for r in range(t):
        for c in range(3):
            cf.data[r * 3 + c] = Float32(tex.coords.at(r, c + 1))
    var coords_py = tensor_to_torch(cf).to(torch.int32).numpy()
    var attrs_py = tensor_to_torch(tex.vl.feats).numpy()
    var origin = Python.list()
    for _ in range(3):
        origin.append(-0.5)
    _ = np.savez(
        path, coords=coords_py, attrs=attrs_py,
        origin=np.array(origin, dtype=np.float32),
        voxel_size=1.0 / Float64(resolution),
    )


def main() raises:
    var image_path = String("")
    var out_prefix = String("output_mojo")
    var seed = 42
    var steps_override = 0
    var with_tex = True
    var cascade_1024 = False
    var remesh = False
    var remesh_band = 1.0
    var remesh_project = 0.9
    var remesh_res = 0  # 0 = pipeline-oppløsningen; lavere = færre triangler
    var i = 1
    while i < len(argv()):
        var a = String(argv()[i])
        if a == "--seed":
            i += 1
            seed = parse_int(String(argv()[i]))
        elif a == "--steps":
            i += 1
            steps_override = parse_int(String(argv()[i]))
        elif a == "--out":
            i += 1
            out_prefix = String(argv()[i])
        elif a == "--no-tex":
            with_tex = False
        elif a == "--remesh":
            remesh = True
        elif a == "--remesh-band":
            i += 1
            remesh_band = parse_float(String(argv()[i]))
        elif a == "--remesh-project":
            i += 1
            remesh_project = parse_float(String(argv()[i]))
        elif a == "--remesh-res":
            i += 1
            remesh_res = parse_int(String(argv()[i]))
        elif a == "--pipeline":
            i += 1
            var pt = String(argv()[i])
            if pt == "1024":
                cascade_1024 = True  # upstream '1024_cascade' (their default)
            elif pt != "512":
                raise Error("--pipeline must be 512 or 1024")
        elif image_path == "":
            image_path = a
        else:
            raise Error("unknown argument: " + a)
        i += 1
    if image_path == "":
        raise Error(
            "usage: mojo run -I . run_image_to_3d.mojo <image.png>"
            " [--seed N] [--steps N] [--out prefix] [--no-tex]"
            " [--pipeline 512|1024] [--remesh]"
            " [--remesh-band B] [--remesh-project P] [--remesh-res N]"
        )

    var t_start = perf_counter_ns()
    var torch = Python.import_module("torch")
    var gpu = gpu_context_from_env()  # WP11: TRELLIS2_GPU=1 -> Metal linear
    var doc = pipeline_config_json()
    var args_node = doc.obj_get(doc.root, "args")
    var ss_params = SamplerParams(doc, args_node, "sparse_structure_sampler", steps_override)
    var shape_params = SamplerParams(doc, args_node, "shape_slat_sampler", steps_override)
    var tex_params = SamplerParams(doc, args_node, "tex_slat_sampler", steps_override)
    var shape_mean = _norm_tensor(doc, args_node, "shape_slat_normalization", "mean")
    var shape_std = _norm_tensor(doc, args_node, "shape_slat_normalization", "std")
    var tex_mean = _norm_tensor(doc, args_node, "tex_slat_normalization", "mean")
    var tex_std = _norm_tensor(doc, args_node, "tex_slat_normalization", "std")
    # pipeline_type '512': cond at 512, ss grid 32, decode resolution 512;
    # '1024' = upstream '1024_cascade': ss grid stays 32, the shape slat
    # cascades 512 -> 1024 and everything downstream runs at 1024
    comptime ss_res = 32
    var resolution = 512
    if cascade_1024:
        resolution = 1024
    print(
        "trellis2-mojo image->3D  (pipeline",
        "1024-cascade" if cascade_1024 else "512", ", seed", seed, end="",
    )
    print(", steps", ss_params.steps, ")")
    print("image:", image_path)

    # conditioning first (upstream constructs DINOv3 pre-seed at
    # from_pretrained; the Mojo loader draws no torch RNG at all), THEN
    # the seed that governs every noise draw
    var t0 = perf_counter_ns()
    var conditioner = ImageConditioner(gpu)
    var cond = conditioner.get_cond(image_path, 512)
    var neg = zeros_like_cond(cond)
    # the 1024 DiTs condition on the 1024-res DINOv3 tokens (4101)
    var cond_hr = Tensor[F32]([1, 1])
    var neg_hr = Tensor[F32]([1, 1])
    if cascade_1024:
        cond_hr = conditioner.get_cond(image_path, 1024)
        neg_hr = zeros_like_cond(cond_hr)
    print("[1/6] conditioning: [", end="")
    print(cond.shape[0], cond.shape[1], cond.shape[2], end="")
    if cascade_1024:
        print("] + [", end="")
        print(cond_hr.shape[0], cond_hr.shape[1], cond_hr.shape[2], end="")
    print("] in", elapsed(t0))
    _ = torch.manual_seed(seed)

    t0 = perf_counter_ns()
    var coords = run_ss_stage(torch, cond, neg, ss_params, ss_res, gpu)
    print("[2/6] sparse structure:", coords.rows, "active voxels @", ss_res, "^3 in", elapsed(t0))
    if coords.rows == 0:
        raise Error("no active voxels — sparse-structure stage produced an empty grid")

    t0 = perf_counter_ns()
    var shape_slat = run_shape_slat_stage(
        torch, "shape_slat_flow_model_512", cond, neg, coords,
        shape_mean, shape_std, shape_params, gpu,
    )
    if cascade_1024:
        # upstream sample_shape_slat_cascade: the LR (512) slat above
        # seeds the subdivision upsample -> 64^3 HR coords -> the 1024
        # DiT resamples the shape slat with the 1024-res conditioning
        # (noise draw order LR -> HR matches upstream's RNG stream)
        var hr_coords = run_cascade_stage(shape_slat, gpu)
        shape_slat = run_shape_slat_stage(
            torch, "shape_slat_flow_model_1024", cond_hr, neg_hr, hr_coords,
            shape_mean, shape_std, shape_params, gpu,
        )
    print("[3/6] shape slat:", shape_slat.coords.rows, "tokens in", elapsed(t0))

    var tex_slat = SparseTensor[F32](Tensor[F32]([1, 1]), IntMatrix(1, 4), 1)
    if with_tex:
        t0 = perf_counter_ns()
        if cascade_1024:
            tex_slat = run_tex_slat_stage(
                torch, "tex_slat_flow_model_1024", cond_hr, neg_hr, shape_slat,
                shape_mean, shape_std, tex_mean, tex_std, tex_params, gpu,
            )
        else:
            tex_slat = run_tex_slat_stage(
                torch, "tex_slat_flow_model_512", cond, neg, shape_slat,
                shape_mean, shape_std, tex_mean, tex_std, tex_params, gpu,
            )
        print("[4/6] tex slat:", tex_slat.coords.rows, "tokens in", elapsed(t0))
    else:
        print("[4/6] tex slat: skipped (--no-tex)")

    t0 = perf_counter_ns()
    var shape_dec = load_unet_decoder("shape_slat_decoder", gpu)
    var decoded = decode_shape(shape_dec, shape_slat)
    print("[5/6] shape decode:", decoded[0].coords.rows, "voxels @", resolution, "^3 in", elapsed(t0))

    t0 = perf_counter_ns()
    var n = decoded[0].coords.rows
    var xyz = IntMatrix(n, 3)
    for r in range(n):
        for c in range(3):
            xyz.set(r, c, decoded[0].coords.at(r, c + 1))
    var aabb_min: List[Float64] = [-0.5, -0.5, -0.5]
    var aabb_max: List[Float64] = [0.5, 0.5, 0.5]
    var mesh = flexible_dual_grid_to_mesh(
        xyz, decoded[0].vl.feats, decoded[1].vl.feats, decoded[2].vl.feats,
        aabb_min, aabb_max, resolution,
    )
    var obj_path = out_prefix + ".obj"
    write_obj(obj_path, mesh[0], mesh[1])
    print("[6/6] mesh:", mesh[0].shape[0], "vertices,", mesh[1].rows, "triangles in", elapsed(t0))
    print("saved:", obj_path)

    if with_tex:
        t0 = perf_counter_ns()
        var tex_dec = load_unet_decoder("tex_slat_decoder", gpu)
        var tex_out = decode_tex(tex_dec, tex_slat, decoded[3])
        var npz_path = out_prefix + "_texvoxels.npz"
        save_tex_voxels(torch, npz_path, tex_out, resolution)
        print("tex decode:", tex_out.coords.rows, "voxels in", elapsed(t0))
        print("saved:", npz_path)

        # WP15 (fase 2, ADR 0008): textured GLB via vertex attributes —
        # the PBR volume is sampled at the mesh vertices in WORLD coords
        # (before the export axis swap, like upstream to_glb), base_color
        # + alpha ride in COLOR_0, metallic/roughness become the global
        # material factors (mean of the sampled values; per-voxel PBR
        # stays in the npz). WP16: micro holes (FDG is non-watertight)
        # are filled first, mirroring upstream's
        # fill_holes(max_hole_perimeter=3e-2) — GLB only, the OBJ/npz
        # stay raw like upstream's MeshWithVoxel payload.
        t0 = perf_counter_ns()
        var fverts = mesh[0].copy()
        var ffaces = IntMatrix(mesh[1].rows, 3)
        for i in range(len(mesh[1].data)):
            ffaces.data[i] = mesh[1].data[i]
        # upstream order (to_glb minus the CUDA-only simplify steps):
        # fill -> repair_non_manifold_edges -> remove_small_connected_
        # components(1e-5) -> fill — rings with a non-manifold edge read
        # as dead-end paths until the corner split turns them into
        # closed loops (100% of dead ends sat on non-manifold edges on
        # the 1024 golden); the corner split also frees small fragment
        # sheets, which upstream then drops before the refill
        if remesh:
            # WP18 (fase 2): upstream's remesh branch — extract the
            # offset surface UDF - eps = 0 around the raw FDG mesh
            # (band = 1 voxel) with narrow-band dual contouring and
            # project 90% back; cracks/holes narrower than ~2 voxels
            # are swallowed by construction and the result is closed
            # with globally consistent winding, so the whole cleanup
            # chain below is unnecessary. Domain like upstream to_glb:
            # center = aabb mean = 0, scale inflated by (R + 3b)/R.
            # --remesh-res decouples the DC grid from the pipeline
            # resolution: triangle count scales ~res^2 and project_back
            # keeps vertices on the true surface, so a lower res is the
            # cheap "smooth it down" knob (band stays in DC-voxel
            # units — 1 voxel @256 swallows what 2 would @512)
            var dc_res = resolution
            if remesh_res > 0:
                dc_res = remesh_res
            var rm_scale = (
                Float64(dc_res) + 3.0 * remesh_band
            ) / Float64(dc_res)
            var rm = remesh_narrow_band_dc(
                fverts, ffaces, 0, 0, 0, rm_scale, dc_res,
                remesh_band, remesh_project,
            )
            fverts = rm[0].copy()
            ffaces = IntMatrix(rm[1].rows, 3)
            for ri in range(len(rm[1].data)):
                ffaces.data[ri] = rm[1].data[ri]
            print(
                "remesh: dual-contoured to", fverts.shape[0], "V /",
                ffaces.rows, "F ( res", dc_res, ", band", remesh_band,
                ", project", remesh_project, ")",
            )
        else:
            var fill_res = fill_small_holes(fverts, ffaces, 3e-2)
            var nv_split = repair_non_manifold_edges(fverts, ffaces)
            var rm_small = remove_small_connected_components(fverts, ffaces, 1e-5)
            var fill2 = fill_small_holes(fverts, ffaces, 3e-2)
            print(
                "holes: filled", fill_res[0], "+", fill2[0],
                "after non-manifold split (", fill_res[1], "->", fill2[1],
                "boundary edges;", nv_split, "verts after split;",
                rm_small[0], "small sheets /", rm_small[1], "faces dropped )",
            )
            # WP16 v7 (own semantics, user decision after the 2026-07-12
            # A/B): weld bit-identical coincident boundary vertices — the
            # leftover closed seam rings are zero-width sheet borders whose
            # cracks and split shading read as dark specks; a fill pass
            # after the weld closes rings that only became closable by it
            var sew = sew_boundary_seams(fverts, ffaces)
            var fill3 = fill_small_holes(fverts, ffaces, 3e-2)
            print(
                "seams: welded", sew[0], "coincident boundary verts (",
                sew[1], "degenerate faces dropped );", fill3[0],
                "components closed post-weld (", fill3[1],
                "boundary edges before )",
            )
        var tv = tex_out.coords.rows
        var tex_xyz = IntMatrix(tv, 3)
        for r in range(tv):
            for c in range(3):
                tex_xyz.set(r, c, tex_out.coords.at(r, c + 1))
        # upstream's LAST cleanup step: unify face orientations (the raw
        # FDG winding is ~50/50 mixed — culled backfaces and cancelled
        # vertex normals look exactly like micro holes); parity-only like
        # cumesh — a global in/out vote was measured useless (the FDG
        # surface is folded in places, see postprocess.mojo). The remesh
        # branch skips it: DC winding is globally consistent already.
        if not remesh:
            var uni = unify_face_orientations(ffaces, fverts)
            print("orient: flipped", uni[0], "of", ffaces.rows, "faces over", uni[1], "sheets")
        var nverts = fverts.shape[0]
        var query = Tensor[F32]([nverts, 3])
        for i in range(nverts * 3):
            # (pos - aabb_min) / voxel_size with aabb [-0.5, 0.5]^3
            query.data[i] = (fverts.data[i] + 0.5) * Float32(resolution)
        var sampled = grid_sample_trilinear(
            tex_out.vl.feats, tex_xyz, resolution, query
        )
        var colors = Tensor[F32]([nverts, 4])
        var msum: Float64 = 0
        var rsum: Float64 = 0
        for i in range(nverts):
            colors.data[i * 4 + 0] = sampled.data[i * 6 + 0]
            colors.data[i * 4 + 1] = sampled.data[i * 6 + 1]
            colors.data[i * 4 + 2] = sampled.data[i * 6 + 2]
            colors.data[i * 4 + 3] = sampled.data[i * 6 + 5]
            msum += Float64(sampled.data[i * 6 + 3])
            rsum += Float64(sampled.data[i * 6 + 4])
        var mf: Float64 = 0
        var rf: Float64 = 1
        if nverts > 0:
            mf = min(max(msum / Float64(nverts), 0.0), 1.0)
            rf = min(max(rsum / Float64(nverts), 0.0), 1.0)
        var gnorms = vertex_normals(fverts, ffaces)
        to_glb_axes(fverts)
        to_glb_axes(gnorms)
        var glb_path = out_prefix + ".glb"
        write_glb(glb_path, fverts, gnorms, colors, ffaces, mf, rf)
        print("glb:", nverts, "vertices with sampled pbr attrs in", elapsed(t0))
        print("saved:", glb_path)

    # structural summary for the trellis-mac comparison (WP0-golden light)
    if mesh[0].shape[0] > 0:
        var lo = List[Float32](length=3, fill=Float32(1e30))
        var hi = List[Float32](length=3, fill=Float32(-1e30))
        for v in range(mesh[0].shape[0]):
            for c in range(3):
                var x = mesh[0].data[v * 3 + c]
                if x < lo[c]:
                    lo[c] = x
                if x > hi[c]:
                    hi[c] = x
        print("bbox min:", lo[0], lo[1], lo[2])
        print("bbox max:", hi[0], hi[1], hi[2])
    print("total:", elapsed(t_start))
