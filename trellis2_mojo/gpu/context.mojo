# WP11: the shared GPU context — one per run. Owns the DeviceContext, the
# grow-only scratches for the linear and attention paths, and the fence
# buffer that is the ONLY reliable commit+wait primitive on 1.0.0b2-Metal
# (ctx.synchronize() does not commit pending work — see gpu/linear.mojo
# for the full list of empirically established laws).
#
# Creation runs a sacrificial cycle + a VERIFIED self-test: the FIRST full
# map-write -> kernel -> map-read cycle in a process delivers corrupt
# reads at 256-byte boundaries (probes 2026-07-10; independent of fences
# and per-kernel warm-up launches — burning one full cycle is the only
# mitigation that held). If the second, verified cycle also misbehaves the
# constructor raises and gpu_context_from_env falls back to CPU.

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import ArcPointer
from std.os import getenv

comptime F32 = DType.float32


def _selftest_scale(
    a: UnsafePointer[Scalar[F32], MutAnyOrigin],
    c: UnsafePointer[Scalar[F32], MutAnyOrigin],
    n: Int,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        c[i] = a[i] * 2.0 + 1.0


struct GpuScratch(Movable):
    """Grow-only device buffers for the linear/conv paths, shared through
    the GpuContext ArcPointer (the fence barrier in each forward makes
    cross-call reuse safe). `e` holds the sparse-conv edge/CSR pack
    (int32: header + row_start + src + kidx). `xs`/`hs`/`bk` are the
    whole-block residency state (WP11 step 10): running x, norm/attn
    scratch and the per-block glue consts (shift/scale/gate/bias pairs)."""

    var a: Optional[DeviceBuffer[F32]]
    var a_cap: Int
    var c: Optional[DeviceBuffer[F32]]
    var c_cap: Int
    var e: Optional[DeviceBuffer[DType.int32]]
    var e_cap: Int
    var xs: Optional[DeviceBuffer[F32]]
    var xs_cap: Int
    var hs: Optional[DeviceBuffer[F32]]
    var hs_cap: Int
    var bk: Optional[DeviceBuffer[F32]]
    var bk_cap: Int

    def __init__(out self):
        self.a = None
        self.a_cap = 0
        self.c = None
        self.c_cap = 0
        self.e = None
        self.e_cap = 0
        self.xs = None
        self.xs_cap = 0
        self.hs = None
        self.hs_cap = 0
        self.bk = None
        self.bk_cap = 0


struct GpuAttnScratch(Movable):
    """Grow-only device buffers for the SDPA composition (gpu/attention.mojo),
    fence-serialized like the linear scratch."""

    var qh: Optional[DeviceBuffer[F32]]   # [H, L, D], scale pre-baked
    var qh_cap: Int
    var kt: Optional[DeviceBuffer[F32]]   # [H, D, Lkv_pad]
    var kt_cap: Int
    var vh: Optional[DeviceBuffer[F32]]   # [H, Lkv_pad, D]
    var vh_cap: Int
    var sc: Optional[DeviceBuffer[F32]]   # [H, L, Lkv_pad] scores/probs
    var sc_cap: Int
    var ob: Optional[DeviceBuffer[F32]]   # [H, L, D]
    var ob_cap: Int
    var su: Optional[DeviceBuffer[F32]]   # [H * L] row sums
    var su_cap: Int
    var ph: Optional[DeviceBuffer[F32]]   # [L, D/2, 2] rope phases (WP11 step 7)
    var ph_cap: Int
    # dedicated cross-kv buffers (WP11 step 10): in the fused block queue
    # the self chain's device-side pack overwrites kt/vh, so the
    # host-packed cross kv needs its own pair
    var ckt: Optional[DeviceBuffer[F32]]  # [H, D, Lkv_pad]
    var ckt_cap: Int
    var cvh: Optional[DeviceBuffer[F32]]  # [H, Lkv_pad, D]
    var cvh_cap: Int

    def __init__(out self):
        self.qh = None
        self.qh_cap = 0
        self.kt = None
        self.kt_cap = 0
        self.vh = None
        self.vh_cap = 0
        self.sc = None
        self.sc_cap = 0
        self.ob = None
        self.ob_cap = 0
        self.su = None
        self.su_cap = 0
        self.ph = None
        self.ph_cap = 0
        self.ckt = None
        self.ckt_cap = 0
        self.cvh = None
        self.cvh_cap = 0


struct GpuContext(Copyable, Movable):
    """One per run: the shared DeviceContext, the shared scratches and the
    fence buffer. Handle-copied into every GpuLinear / attention call site
    at model-load time."""

    var ctx: DeviceContext
    var scratch: ArcPointer[GpuScratch]
    var attn: ArcPointer[GpuAttnScratch]
    var fence: DeviceBuffer[F32]
    # WP19: TRELLIS2_GPU_F16=1 -> vekt-GEMM-ene kjører f16-shared-fliser
    # (A castes til f16 på shared-fyllet — ulp-klasse-numerikk; +32-40 %
    # målt i microbench_gpu_gemm); default av = bit-eksakt som før
    var f16: Bool

    def __init__(out self) raises:
        self.f16 = getenv("TRELLIS2_GPU_F16") == "1"
        self.ctx = DeviceContext()
        self.scratch = ArcPointer(GpuScratch())
        self.attn = ArcPointer(GpuAttnScratch())
        self.fence = self.ctx.enqueue_create_buffer[F32](1)
        with self.fence.map_to_host() as h:
            h[0] = 0  # host-write: makes the first barrier map commit
        # sacrificial cycle + verified self-test (see file header)
        var sin = self.ctx.enqueue_create_buffer[F32](1024)
        var sout = self.ctx.enqueue_create_buffer[F32](1024)
        for cycle in range(2):
            with sin.map_to_host() as h:
                for i in range(1024):
                    h[i] = Float32(i % 51) + Float32(cycle)
            self.ctx.enqueue_function[_selftest_scale](
                sin.unsafe_ptr(), sout.unsafe_ptr(), 1024,
                grid_dim=(4,), block_dim=(256,),
            )
            self.barrier()
            if cycle == 1:
                with sout.map_to_host() as h:
                    for i in range(1024):
                        var expct = (Float32(i % 51) + 1.0) * 2.0 + 1.0
                        if abs(h[i] - expct) > 1e-6:
                            raise Error("GpuContext self-test failed at " + String(i))

    def barrier(self) raises:
        """Commit all pending queue work (h2d blits, kernels, d2h copies)
        and wait for it. The re-write keeps the buffer host-dirty so the
        NEXT barrier commits too."""
        with self.fence.map_to_host() as h:
            h[0] = h[0] + 1


def gpu_context_from_env() -> Optional[GpuContext]:
    """TRELLIS2_GPU=1 enables the GPU paths; anything else (or a machine
    without a usable GPU, or a failed self-test) falls back to CPU."""
    if getenv("TRELLIS2_GPU") != "1":
        return None
    try:
        var g = GpuContext()
        if g.f16:
            print("[gpu] offload enabled (api: " + String(g.ctx.api()) + ", f16 weight-gemm)")
        else:
            print("[gpu] offload enabled (api: " + String(g.ctx.api()) + ")")
        return g^
    except:
        print("[gpu] TRELLIS2_GPU=1 but no usable device / self-test failed — CPU fallback")
        return None
