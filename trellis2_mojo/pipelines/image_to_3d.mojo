# Mojo side of the Trellis2ImageTo3DPipeline core (WP9): everything between
# "conditioning is ready" and "mesh features are ready" — the stages of
# pipelines/trellis2_image_to_3d.py that run models:
#
#   sample_sparse_structure: FlowEuler(ss_flow) -> ss_decoder -> occupancy
#     threshold (+ optional max-pool downscale) -> active voxel coords.
#   sample_slat: FlowEuler(slat_flow) on fixed coords -> mean/std
#     de-normalization (sample_shape_slat / sample_tex_slat).
#   decode_shape: SparseUnetVaeDecoder (pred_subdiv) + fdg_head.
#
# The velocity adapters below plug the WP8 models into the WP2 sampler:
# the sampler's math is shape-agnostic, so sparse sampling reuses the dense
# loop with feats as the state and coords/cond carried by the adapter
# (coords never change during a trajectory).
#
# Deliberately still Python (per ADR 0001/0004, wired by the runner, not
# here): image conditioning (DINO), rembg preprocessing, noise generation,
# safetensors loading, flexible_dual_grid_to_mesh and GLB export.

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix, stable_argsort
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.samplers.flow_euler import VelocityModel, FlowEulerSampler
from trellis2_mojo.models.sparse_structure_flow import SparseStructureFlowModel
from trellis2_mojo.models.sparse_structure_vae import SparseStructureDecoder
from trellis2_mojo.models.structured_latent_flow import SLatFlowModel
from trellis2_mojo.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
from trellis2_mojo.models.sc_vaes.fdg_vae import fdg_head

comptime F32 = DType.float32


struct SSFlowVelocity(Copyable, Movable, VelocityModel):
    """VelocityModel adapter: dense SS flow model + its (neg_)cond."""

    var model: SparseStructureFlowModel
    var cond: Tensor[F32]      # [N, Lc, Cc]
    var neg_cond: Tensor[F32]

    def __init__(
        out self,
        var model: SparseStructureFlowModel,
        var cond: Tensor[F32],
        var neg_cond: Tensor[F32],
    ):
        self.model = model^
        self.cond = cond^
        self.neg_cond = neg_cond^

    def predict(self, x_t: Tensor[F32], t1000: Float64, use_neg_cond: Bool) raises -> Tensor[F32]:
        var t = Tensor[F32]([x_t.shape[0]], Float32(t1000))
        if use_neg_cond:
            return self.model.forward(x_t, t, self.neg_cond)
        return self.model.forward(x_t, t, self.cond)


struct SlatFlowVelocity(Copyable, Movable, VelocityModel):
    """VelocityModel adapter for the sparse flow: the sampler state is the
    feats tensor [T, C]; coords (fixed for the whole trajectory), cond and
    the optional concat_cond (texture path) live here."""

    var model: SLatFlowModel
    var coords: IntMatrix
    var batch_size: Int
    var cond: Tensor[F32]
    var neg_cond: Tensor[F32]
    var has_concat: Bool
    var concat_feats: Tensor[F32]

    def __init__(
        out self,
        var model: SLatFlowModel,
        var coords: IntMatrix,
        var cond: Tensor[F32],
        var neg_cond: Tensor[F32],
    ) raises:
        self.model = model^
        self.batch_size = coords.col_max(0) + 1
        self.coords = coords^
        self.cond = cond^
        self.neg_cond = neg_cond^
        self.has_concat = False
        self.concat_feats = Tensor[F32]([1])

    def set_concat(mut self, var concat_feats: Tensor[F32]) raises:
        """Texture path: sparse_cat([x_t, concat_cond]) before every model
        call (the concat rides on the same coords)."""
        self.has_concat = True
        self.concat_feats = concat_feats^

    def predict(self, x_t: Tensor[F32], t1000: Float64, use_neg_cond: Bool) raises -> Tensor[F32]:
        var sp = SparseTensor[F32](x_t.copy(), self.coords.copy(), self.batch_size)
        var t = Tensor[F32]([self.batch_size], Float32(t1000))
        var out: SparseTensor[F32]
        if self.has_concat:
            var cc = SparseTensor[F32](self.concat_feats.copy(), self.coords.copy(), self.batch_size)
            if use_neg_cond:
                out = self.model.forward(sp, t, self.neg_cond, cc)
            else:
                out = self.model.forward(sp, t, self.cond, cc)
        else:
            if use_neg_cond:
                out = self.model.forward(sp, t, self.neg_cond)
            else:
                out = self.model.forward(sp, t, self.cond)
        return out.vl.feats.copy()


def occupancy_to_coords(occ: Tensor[F32], resolution: Int) raises -> IntMatrix:
    """decoder(z) > 0 -> (optional max-pool downscale) -> argwhere coords
    (n, x, y, z), in lexicographic order over [N, 1, X, Y, Z]."""
    if len(occ.shape) != 5 or occ.shape[1] != 1:
        raise Error("occupancy_to_coords: expected [N, 1, X, Y, Z]")
    var n = occ.shape[0]
    var r = occ.shape[2]
    var ratio = 1
    if r != resolution:
        # F.max_pool3d(bool, ratio, ratio) > 0.5 == any() over the window
        ratio = r // resolution
        if ratio * resolution != r:
            raise Error("occupancy_to_coords: resolution must divide decoder output")
    var rr = r // ratio

    var active = List[Bool](length=n * rr * rr * rr, fill=False)
    for b in range(n):
        for x in range(r):
            for y in range(r):
                for z in range(r):
                    if occ.data[((b * r + x) * r + y) * r + z] > 0:
                        active[((b * rr + x // ratio) * rr + y // ratio) * rr + z // ratio] = True

    var count = 0
    for i in range(len(active)):
        if active[i]:
            count += 1
    var coords = IntMatrix(count, 4)
    var e = 0
    for b in range(n):
        for x in range(rr):
            for y in range(rr):
                for z in range(rr):
                    if active[((b * rr + x) * rr + y) * rr + z]:
                        coords.set(e, 0, b)
                        coords.set(e, 1, x)
                        coords.set(e, 2, y)
                        coords.set(e, 3, z)
                        e += 1
    return coords^


def sample_sparse_structure(
    sampler: FlowEulerSampler,
    velocity: SSFlowVelocity,
    decoder: SparseStructureDecoder,
    noise: Tensor[F32],
    resolution: Int,
    steps: Int,
    rescale_t: Float64,
    guidance_strength: Float64,
    interval_lo: Float64,
    interval_hi: Float64,
    guidance_rescale: Float64 = 0.0,
) raises -> IntMatrix:
    """Pipeline.sample_sparse_structure: latent sampling + decode +
    thresholding to active voxel coords. Noise comes from the caller's native
    Mojo RNG so trajectories are reproducible. The real
    pipeline.json runs guidance_rescale 0.7 here (dense std semantics)."""
    var z = sampler.sample_cfg_interval(
        velocity, noise, steps, rescale_t, guidance_strength, interval_lo, interval_hi,
        guidance_rescale,
    ).samples.copy()
    return occupancy_to_coords(decoder.forward(z), resolution)


def sample_slat(
    sampler: FlowEulerSampler,
    velocity: SlatFlowVelocity,
    noise_feats: Tensor[F32],
    mean: Tensor[F32],
    std: Tensor[F32],
    steps: Int,
    rescale_t: Float64,
    guidance_strength: Float64,
    interval_lo: Float64,
    interval_hi: Float64,
    guidance_rescale: Float64 = 0.0,
) raises -> SparseTensor[F32]:
    """Pipeline.sample_shape_slat / sample_tex_slat: FlowEuler over feats on
    fixed coords, then slat * std + mean (per-channel de-normalization).
    The real pipeline.json runs guidance_rescale 0.5 on the shape stage —
    per-segment VarLen std semantics, so the sampler gets the offsets."""
    var seg_offsets = SparseTensor[F32]._cal_offsets(velocity.coords, velocity.batch_size)
    var feats = sampler.sample_cfg_interval(
        velocity, noise_feats, steps, rescale_t, guidance_strength, interval_lo, interval_hi,
        guidance_rescale, seg_offsets,
    ).samples.copy()
    var c = feats.shape[1]
    if mean.numel() != c or std.numel() != c:
        raise Error("sample_slat: mean/std channel mismatch")
    var out = Tensor[F32](feats.shape)
    for row in range(feats.shape[0]):
        for ci in range(c):
            out.data[row * c + ci] = feats.data[row * c + ci] * std.data[ci] + mean.data[ci]
    return SparseTensor[F32](out^, velocity.coords.copy(), velocity.batch_size)


def decode_shape(
    decoder: SparseUnetVaeDecoder, slat: SparseTensor[F32], voxel_margin: Float64 = 0.5
) raises -> Tuple[
    SparseTensor[F32], SparseTensor[F32], SparseTensor[F32], List[SparseTensor[F32]]
]:
    """Pipeline.decode_shape_slat minus the Python mesh extraction:
    -> (vertices, intersected, quad_lerp, subs). The subs guide the texture
    decoder; the three heads feed flexible_dual_grid_to_mesh in Python."""
    var pair = decoder.forward(slat)
    var heads = fdg_head(pair[0], voxel_margin)
    return (heads[0].copy(), heads[1].copy(), heads[2].copy(), pair[1].copy())


def cascade_coords(
    hr_coords: IntMatrix, lr_resolution: Int, resolution: Int, max_num_tokens: Int
) raises -> Tuple[IntMatrix, Int]:
    """The quantize/unique/token-budget loop of sample_shape_slat_cascade:
    quant = int((xyz + 0.5) / lr_resolution * (hr_resolution / 16)), rows
    deduplicated and sorted lexicographically; hr_resolution drops by 128
    until the token count fits (or the 1024 floor is hit).
    -> (coords, hr_resolution)."""
    var n = hr_coords.rows
    var hr_resolution = resolution
    while True:
        var k = hr_resolution // 16
        # quantize, then encode rows as single keys for sort + dedupe
        var qs = List[Int]()
        var qmax = 0
        for r_i in range(n):
            for c in range(1, 4):
                var q = Int(
                    (Float32(hr_coords.at(r_i, c)) + 0.5)
                    / Float32(lr_resolution) * Float32(k)
                )
                qs.append(q)
                if q > qmax:
                    qmax = q
        var m = qmax + 1
        var keys = List[Int]()
        for r_i in range(n):
            var key = hr_coords.at(r_i, 0)
            for c in range(3):
                key = key * m + qs[r_i * 3 + c]
            keys.append(key)
        var order = stable_argsort(keys)
        var num_tokens = 0
        for i in range(n):
            if i == 0 or keys[order[i]] != keys[order[i - 1]]:
                num_tokens += 1
        if num_tokens < max_num_tokens or hr_resolution == 1024:
            var coords = IntMatrix(num_tokens, 4)
            var e = 0
            for i in range(n):
                if i > 0 and keys[order[i]] == keys[order[i - 1]]:
                    continue
                var key = keys[order[i]]
                coords.set(e, 3, key % m)
                key //= m
                coords.set(e, 2, key % m)
                key //= m
                coords.set(e, 1, key % m)
                coords.set(e, 0, key // m)
                e += 1
            return (coords^, hr_resolution)
        hr_resolution -= 128


def normalize_slat(slat: SparseTensor[F32], mean: Tensor[F32], std: Tensor[F32]) raises -> SparseTensor[F32]:
    """(slat - mean) / std — sample_tex_slat normalizes the shape slat with
    the SHAPE normalization before concatenation."""
    var c = slat.vl.feats.shape[1]
    if mean.numel() != c or std.numel() != c:
        raise Error("normalize_slat: mean/std channel mismatch")
    var out = Tensor[F32](slat.vl.feats.shape)
    for row in range(slat.vl.feats.shape[0]):
        for ci in range(c):
            out.data[row * c + ci] = (
                slat.vl.feats.data[row * c + ci] - mean.data[ci]
            ) / std.data[ci]
    return slat.replace(out^)


def decode_tex(
    decoder: SparseUnetVaeDecoder, slat: SparseTensor[F32], guide_subs: List[SparseTensor[F32]]
) raises -> SparseTensor[F32]:
    """Pipeline.decode_tex_slat: guided decode (subs from the shape decoder)
    followed by * 0.5 + 0.5 into color space."""
    var h = decoder.forward_guided(slat, guide_subs)
    var out = Tensor[F32](h.vl.feats.shape)
    for i in range(h.vl.feats.numel()):
        out.data[i] = h.vl.feats.data[i] * 0.5 + 0.5
    return h.replace(out^)
