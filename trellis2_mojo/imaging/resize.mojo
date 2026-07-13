# PIL-exact Lanczos resize on 8-bit images (WP14). Mirrors Pillow's
# Resample.c 8bpc path bit-for-bit:
#   - separable lanczos (a=3), support scaled by in/out ratio on downscale
#   - per-output-pixel coefficient windows computed in f64, then quantized
#     to integers at PRECISION_BITS = 22 with round-half-away-from-zero
#     (C truncation of v +/- 0.5)
#   - horizontal pass first, then vertical, with clip8 u8 quantization
#     BETWEEN the passes (this inter-pass rounding is why a float
#     implementation can't match PIL)
#   - accumulator starts at the 1 << 21 rounding bias; clip8 caps at
#     2^30 and shifts down by 22
# Pillow's temp image trims rows the vertical pass never reads
# (ybox_first/last) — a pure work optimization; processing all rows gives
# identical output. Coefficient sums stay < 2^31 (lanczos ringing < 1.3x),
# so Mojo's 64-bit Int matches Pillow's INT32 exactly.
#
# RGBA: PIL's Image.resize converts RGBA -> RGBa (premultiplied, the
# MULDIV255 +128-bias divide), resamples the premultiplied bands, and
# converts back with TRUNCATING division (255*v/alpha, alpha 0/255 pass
# through) — verified empirically against Pillow 12.3; skipping this
# round-trip mismatches ~70% of pixels.
#
# Parity: tests/parity/parity_wp14_vs_torch.mojo — BIT-identical to
# PIL.Image.resize(LANCZOS) across up/down/non-square/RGB/RGBA cases.

from std.math import ceil, floor, sin

from trellis2_mojo.io.image import ImageU8

comptime PRECISION_BITS = 22
comptime PI = 3.141592653589793


def _sinc(x: Float64) raises -> Float64:
    if x == 0.0:
        return 1.0
    var y = x * PI
    return sin(y) / y


def _lanczos(x: Float64) raises -> Float64:
    if -3.0 <= x and x < 3.0:
        return _sinc(x) * _sinc(x / 3.0)
    return 0.0


def _clip8(v: Int) -> UInt8:
    if v >= (1 << (PRECISION_BITS + 8)):
        return 255
    if v <= 0:
        return 0
    return UInt8(v >> PRECISION_BITS)


def _coeffs(in_size: Int, out_size: Int) raises -> Tuple[List[Int], List[Int], Int]:
    """Pillow precompute_coeffs + normalize_coeffs_8bpc for the full-image
    box: -> (bounds [out*2: xmin, xmax], int coeffs [out*ksize], ksize)."""
    var scale = Float64(in_size) / Float64(out_size)
    var filterscale = scale
    if filterscale < 1.0:
        filterscale = 1.0
    var support = 3.0 * filterscale
    var ksize = Int(ceil(support)) * 2 + 1
    var bounds = List[Int](length=out_size * 2, fill=0)
    var kk = List[Float64](length=out_size * ksize, fill=0)
    var ss = 1.0 / filterscale
    for xx in range(out_size):
        var center = (Float64(xx) + 0.5) * scale
        # C (int)-casts truncate toward zero; after the clamps that's
        # equivalent to floor here (see xmin note in Resample.c)
        var xmin = Int(floor(center - support + 0.5))
        if xmin < 0:
            xmin = 0
        var xmax = Int(floor(center + support + 0.5))
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        var ww = 0.0
        for x in range(xmax):
            var w = _lanczos((Float64(x + xmin) - center + 0.5) * ss)
            kk[xx * ksize + x] = w
            ww += w
        if ww != 0.0:
            for x in range(xmax):
                kk[xx * ksize + x] /= ww
        bounds[xx * 2] = xmin
        bounds[xx * 2 + 1] = xmax
    var ki = List[Int](length=out_size * ksize, fill=0)
    for i in range(out_size * ksize):
        var v = kk[i] * Float64(1 << PRECISION_BITS)
        if v < 0:
            ki[i] = Int(ceil(v - 0.5))  # C trunc of (-0.5 + v), v negative
        else:
            ki[i] = Int(floor(v + 0.5))
    return (bounds^, ki^, ksize)


def _muldiv255(v: Int, a: Int) -> Int:
    var tmp = v * a + 128
    return ((tmp >> 8) + tmp) >> 8


def _premultiply(img: ImageU8) raises -> ImageU8:
    """PIL convert RGBA -> RGBa (Convert.c rgba2rgbA, MULDIV255)."""
    var out = ImageU8(img.width, img.height, 4)
    for i in range(img.width * img.height):
        var a = Int(img.data[i * 4 + 3])
        for cc in range(3):
            out.data[i * 4 + cc] = UInt8(_muldiv255(Int(img.data[i * 4 + cc]), a))
        out.data[i * 4 + 3] = img.data[i * 4 + 3]
    return out^


def _unpremultiply(img: ImageU8) raises -> ImageU8:
    """PIL convert RGBa -> RGBA (Convert.c rgbA2rgba): truncating
    255*v/alpha capped at 255; alpha 0/255 passes through."""
    var out = ImageU8(img.width, img.height, 4)
    for i in range(img.width * img.height):
        var a = Int(img.data[i * 4 + 3])
        for cc in range(3):
            var v = Int(img.data[i * 4 + cc])
            if a != 0 and a != 255:
                v = (255 * v) // a
                if v > 255:
                    v = 255
            out.data[i * 4 + cc] = UInt8(v)
        out.data[i * 4 + 3] = img.data[i * 4 + 3]
    return out^


def resize_lanczos(img: ImageU8, out_w: Int, out_h: Int) raises -> ImageU8:
    """PIL Image.resize((out_w, out_h), LANCZOS), bit-identical (incl. the
    RGBa premultiply round-trip PIL applies to RGBA input)."""
    if out_w == img.width and out_h == img.height:
        return img.copy()  # PIL short-circuits to copy() BEFORE the RGBa round-trip
    if img.channels == 4:
        return _unpremultiply(_resize_core(_premultiply(img), out_w, out_h))
    return _resize_core(img, out_w, out_h)


def _resize_core(img: ImageU8, out_w: Int, out_h: Int) raises -> ImageU8:
    var c = img.channels
    var src = img.copy()
    if out_w != img.width:
        var t = _coeffs(img.width, out_w)
        var tmp = ImageU8(out_w, src.height, c)
        for y in range(src.height):
            for xx in range(out_w):
                var xmin = t[0][xx * 2]
                var xmax = t[0][xx * 2 + 1]
                for b in range(c):
                    var acc = 1 << (PRECISION_BITS - 1)
                    for x in range(xmax):
                        acc += Int(src.data[(y * src.width + xmin + x) * c + b]) * t[1][xx * t[2] + x]
                    tmp.data[(y * out_w + xx) * c + b] = _clip8(acc)
        src = tmp^
    if out_h != src.height:
        var t = _coeffs(src.height, out_h)
        var out = ImageU8(src.width, out_h, c)
        for yy in range(out_h):
            var ymin = t[0][yy * 2]
            var ymax = t[0][yy * 2 + 1]
            for x in range(src.width):
                for b in range(c):
                    var acc = 1 << (PRECISION_BITS - 1)
                    for y in range(ymax):
                        acc += Int(src.data[((ymin + y) * src.width + x) * c + b]) * t[1][yy * t[2] + y]
                    out.data[(yy * src.width + x) * c + b] = _clip8(acc)
        src = out^
    return src^
