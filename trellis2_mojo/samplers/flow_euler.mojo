from std.math import sqrt
# Mojo port of trellis2/pipelines/samplers/flow_euler.py (+ the CFG and
# guidance-interval mixins).
#
# The Python original composes behavior through mixin MRO
# (GuidanceInterval -> ClassifierFreeGuidance -> FlowEulerSampler). Mojo has
# no inheritance, so the chain is collapsed into one struct with a guidance
# mode: sample() / sample_cfg() / sample_cfg_interval() correspond to
# FlowEulerSampler / FlowEulerCfgSampler / FlowEulerGuidanceIntervalSampler.
#
# The model is abstracted as the VelocityModel trait so the sampler runs
# against native Mojo models through the VelocityModel trait.
#
# guidance_rescale (classifier_free_guidance_mixin.py's CFG rescale) IS
# used by the real TRELLIS.2-4B pipeline.json (ss: 0.7, shape-slat: 0.5) and
# is ported. The source has two std semantics, mirrored here: dense tensors
# use unbiased std per dim-0 row over the remaining dims;
# VarLen/sparse state (seg_offsets given) uses VarLenTensor.std —
# sqrt(mean(x^2) - mean(x)^2), biased, per segment over tokens x channels.
#
# v1 operates on dense Tensor[F32] (the sparse-structure flow model is
# dense). The SparseTensor variant reuses the same math on feats and comes
# with WP8.

from trellis2_mojo.sparse.tensor import Tensor, OP_ADD, OP_SUB, OP_MUL

comptime F32 = DType.float32

comptime GUIDANCE_NONE = 0
comptime GUIDANCE_CFG = 1
comptime GUIDANCE_CFG_INTERVAL = 2


trait VelocityModel:
    """A flow-matching velocity model: v = model(x_t, 1000*t, cond).
    The implementation owns its conditioning; use_neg_cond selects the
    negative conditioning for classifier-free guidance."""

    def predict(self, x_t: Tensor[F32], t1000: Float64, use_neg_cond: Bool) raises -> Tensor[F32]:
        ...


struct SampleResult(Copyable, Movable):
    """Mirror of the edict returned by FlowEulerSampler.sample."""

    var samples: Tensor[F32]
    var pred_x_t: List[Tensor[F32]]
    var pred_x_0: List[Tensor[F32]]

    def __init__(out self, var samples: Tensor[F32]):
        self.samples = samples^
        self.pred_x_t = List[Tensor[F32]]()
        self.pred_x_0 = List[Tensor[F32]]()


def _scaled(t: Tensor[F32], s: Float64) raises -> Tensor[F32]:
    return t._binop_scalar(Float32(s), OP_MUL)


struct FlowEulerSampler(Copyable, Movable):
    var sigma_min: Float64

    def __init__(out self, sigma_min: Float64):
        self.sigma_min = sigma_min

    # x_0 = (1 - sigma_min) * x_t - (sigma_min + (1 - sigma_min) * t) * v
    # eps = (1 - t) * v + x_t
    def _v_to_xstart(self, x_t: Tensor[F32], t: Float64, v: Tensor[F32]) raises -> Tensor[F32]:
        var c = self.sigma_min + (1.0 - self.sigma_min) * t
        return _scaled(x_t, 1.0 - self.sigma_min)._binop_flat(_scaled(v, c), OP_SUB)

    def _v_to_eps(self, x_t: Tensor[F32], t: Float64, v: Tensor[F32]) raises -> Tensor[F32]:
        return _scaled(v, 1.0 - t)._binop_flat(x_t, OP_ADD)

    def _rescale_pred(
        self,
        x_t: Tensor[F32],
        t: Float64,
        pred: Tensor[F32],
        pred_pos: Tensor[F32],
        guidance_rescale: Float64,
        seg_offsets: List[Int],
    ) raises -> Tensor[F32]:
        """CFG rescale (classifier_free_guidance_mixin.py): match the cfg
        prediction's x0-std to the positive branch's, then blend and map
        back to velocity space. Dense x_t (seg_offsets empty): unbiased std —
        unbiased, one std per dim-0 row over the remaining dims. VarLen x_t
        (seg_offsets = token offsets): VarLenTensor.std —
        sqrt(mean(x^2) - mean(x)^2), biased, one std per segment."""
        var x0_pos = self._v_to_xstart(x_t, t, pred_pos)
        var x0_cfg = self._v_to_xstart(x_t, t, pred)
        var rs = x_t.row_size()
        var varlen = len(seg_offsets) > 0
        var bounds = List[Int]()
        if varlen:
            for i in range(len(seg_offsets)):
                bounds.append(seg_offsets[i] * rs)
        else:
            for r in range(x_t.rows() + 1):
                bounds.append(r * rs)
        var out = Tensor[F32](x_t.shape)
        var gr = Float32(guidance_rescale)
        var a = Float32(1.0 - self.sigma_min)
        var cc = Float32(self.sigma_min + (1.0 - self.sigma_min) * t)
        for b in range(len(bounds) - 1):
            var lo = bounds[b]
            var hi = bounds[b + 1]
            var m = hi - lo
            if m == 0:
                continue
            var std_pos: Float32
            var std_cfg: Float32
            if varlen:
                var s1: Float32 = 0
                var s2: Float32 = 0
                var c1: Float32 = 0
                var c2: Float32 = 0
                for e in range(lo, hi):
                    s1 += x0_pos.data[e]
                    s2 += x0_pos.data[e] * x0_pos.data[e]
                    c1 += x0_cfg.data[e]
                    c2 += x0_cfg.data[e] * x0_cfg.data[e]
                var mp = s1 / Float32(m)
                var mc = c1 / Float32(m)
                std_pos = sqrt(s2 / Float32(m) - mp * mp)
                std_cfg = sqrt(c2 / Float32(m) - mc * mc)
            else:
                var s1: Float32 = 0
                var c1: Float32 = 0
                for e in range(lo, hi):
                    s1 += x0_pos.data[e]
                    c1 += x0_cfg.data[e]
                var mp = s1 / Float32(m)
                var mc = c1 / Float32(m)
                var vp: Float32 = 0
                var vc: Float32 = 0
                for e in range(lo, hi):
                    var dp = x0_pos.data[e] - mp
                    var dc = x0_cfg.data[e] - mc
                    vp += dp * dp
                    vc += dc * dc
                std_pos = sqrt(vp / Float32(m - 1))
                std_cfg = sqrt(vc / Float32(m - 1))
            var ratio = std_pos / std_cfg
            for e in range(lo, hi):
                var x0r = x0_cfg.data[e] * ratio
                var x0 = gr * x0r + (1.0 - gr) * x0_cfg.data[e]
                out.data[e] = (a * x_t.data[e] - x0) / cc
        return out^

    def _inference[M: VelocityModel](
        self,
        model: M,
        x_t: Tensor[F32],
        t: Float64,
        mode: Int,
        guidance_strength: Float64,
        interval_lo: Float64,
        interval_hi: Float64,
        guidance_rescale: Float64,
        seg_offsets: List[Int],
    ) raises -> Tensor[F32]:
        var use_cfg = mode == GUIDANCE_CFG
        if mode == GUIDANCE_CFG_INTERVAL:
            # GuidanceIntervalSamplerMixin: CFG inside [lo, hi], plain outside
            use_cfg = interval_lo <= t and t <= interval_hi
        if not use_cfg or guidance_strength == 1.0:
            return model.predict(x_t, 1000.0 * t, False)
        if guidance_strength == 0.0:
            return model.predict(x_t, 1000.0 * t, True)
        var pred_pos = model.predict(x_t, 1000.0 * t, False)
        var pred_neg = model.predict(x_t, 1000.0 * t, True)
        # guidance_strength * pos + (1 - guidance_strength) * neg
        var pred = _scaled(pred_pos, guidance_strength)._binop_flat(
            _scaled(pred_neg, 1.0 - guidance_strength), OP_ADD
        )
        if guidance_rescale > 0.0:
            return self._rescale_pred(x_t, t, pred, pred_pos, guidance_rescale, seg_offsets)
        return pred^

    def _sample_impl[M: VelocityModel](
        self,
        model: M,
        noise: Tensor[F32],
        steps: Int,
        rescale_t: Float64,
        mode: Int,
        guidance_strength: Float64,
        interval_lo: Float64,
        interval_hi: Float64,
        guidance_rescale: Float64,
        seg_offsets: List[Int],
    ) raises -> SampleResult:
        var sample = noise.copy()
        var t_seq = List[Float64]()
        for i in range(steps + 1):
            var t = 1.0 - Float64(i) / Float64(steps)  # np.linspace(1, 0, steps+1)
            t_seq.append(rescale_t * t / (1.0 + (rescale_t - 1.0) * t))

        var ret = SampleResult(noise.copy())
        for i in range(steps):
            var t = t_seq[i]
            var t_prev = t_seq[i + 1]
            var pred_v = self._inference(
                model, sample, t, mode, guidance_strength, interval_lo, interval_hi,
                guidance_rescale, seg_offsets,
            )
            var pred_x_0 = self._v_to_xstart(sample, t, pred_v)
            # pred_x_prev = x_t - (t - t_prev) * v
            sample = sample._binop_flat(_scaled(pred_v, t - t_prev), OP_SUB)
            ret.pred_x_t.append(sample.copy())
            ret.pred_x_0.append(pred_x_0^)
        ret.samples = sample^
        return ret^

    def sample[M: VelocityModel](
        self, model: M, noise: Tensor[F32], steps: Int = 50, rescale_t: Float64 = 1.0
    ) raises -> SampleResult:
        """FlowEulerSampler.sample (no guidance)."""
        return self._sample_impl(
            model, noise, steps, rescale_t, GUIDANCE_NONE, 1.0, 0.0, 1.0, 0.0, List[Int]()
        )

    def sample_cfg[M: VelocityModel](
        self,
        model: M,
        noise: Tensor[F32],
        steps: Int = 50,
        rescale_t: Float64 = 1.0,
        guidance_strength: Float64 = 3.0,
        guidance_rescale: Float64 = 0.0,
        seg_offsets: List[Int] = List[Int](),
    ) raises -> SampleResult:
        """FlowEulerCfgSampler.sample."""
        return self._sample_impl(
            model, noise, steps, rescale_t, GUIDANCE_CFG, guidance_strength, 0.0, 1.0,
            guidance_rescale, seg_offsets,
        )

    def sample_cfg_interval[M: VelocityModel](
        self,
        model: M,
        noise: Tensor[F32],
        steps: Int = 50,
        rescale_t: Float64 = 1.0,
        guidance_strength: Float64 = 3.0,
        interval_lo: Float64 = 0.0,
        interval_hi: Float64 = 1.0,
        guidance_rescale: Float64 = 0.0,
        seg_offsets: List[Int] = List[Int](),
    ) raises -> SampleResult:
        """FlowEulerGuidanceIntervalSampler.sample. seg_offsets marks VarLen
        state (feats [T, C] + token offsets) so CFG rescale uses per-segment
        VarLenTensor.std semantics instead of dense per-row unbiased std."""
        return self._sample_impl(
            model,
            noise,
            steps,
            rescale_t,
            GUIDANCE_CFG_INTERVAL,
            guidance_strength,
            interval_lo,
            interval_hi,
            guidance_rescale,
            seg_offsets,
        )
