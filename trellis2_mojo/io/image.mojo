# Netpbm readers in pure Mojo (WP14): PAM P7 (the runner's RGBA input
# format — ADR 0007 requires a real alpha channel) and PPM P6 (RGB, for
# completeness/tests). 8-bit only (MAXVAL 255), binary rasters.
#
# PNG stays a documented conversion step (README_MOJO) — writing P7 from
# any PNG is a PIL one-liner; a pure-Mojo PNG decoder is its own decision
# point per ADR 0007.

from trellis2_mojo.io.hf_cache import read_file_bytes


struct ImageU8(Copyable, Movable):
    """8-bit image, rows top-down, channels interleaved (H x W x C)."""

    var width: Int
    var height: Int
    var channels: Int
    var data: List[UInt8]

    def __init__(out self, width: Int, height: Int, channels: Int) raises:
        self.width = width
        self.height = height
        self.channels = channels
        self.data = List[UInt8](length=width * height * channels, fill=0)

    def at(self, y: Int, x: Int, c: Int) raises -> UInt8:
        return self.data[(y * self.width + x) * self.channels + c]


def _is_space(b: UInt8) -> Bool:
    return b == 0x20 or b == 0x09 or b == 0x0A or b == 0x0D or b == 0x0B or b == 0x0C


def _skip_ws_and_comments(bytes: List[UInt8], mut pos: Int) raises:
    """Whitespace and '#' comments (to end of line) between header tokens."""
    while pos < len(bytes):
        if _is_space(bytes[pos]):
            pos += 1
        elif bytes[pos] == 0x23:  # '#'
            while pos < len(bytes) and bytes[pos] != 0x0A:
                pos += 1
        else:
            return


def _read_token(bytes: List[UInt8], mut pos: Int) raises -> String:
    _skip_ws_and_comments(bytes, pos)
    var out = List[UInt8]()
    while pos < len(bytes) and not _is_space(bytes[pos]):
        out.append(bytes[pos])
        pos += 1
    if len(out) == 0:
        raise Error("netpbm: unexpected end of header")
    return String(from_utf8=Span(out))


def _read_int(bytes: List[UInt8], mut pos: Int) raises -> Int:
    var tok = _read_token(bytes, pos)
    var b = tok.as_bytes()
    var v = 0
    for i in range(len(b)):
        if b[i] < 0x30 or b[i] > 0x39:
            raise Error("netpbm: expected integer, got " + tok)
        v = v * 10 + Int(b[i] - 0x30)
    return v


def _copy_raster(bytes: List[UInt8], start: Int, mut img: ImageU8) raises:
    var n = img.width * img.height * img.channels
    if start + n > len(bytes):
        raise Error("netpbm: raster truncated")
    for i in range(n):
        img.data[i] = bytes[start + i]


def read_pam(path: String) raises -> ImageU8:
    """PAM P7, DEPTH 3 (RGB) or 4 (RGB_ALPHA), MAXVAL 255."""
    var bytes = read_file_bytes(path)
    var pos = 0
    if _read_token(bytes, pos) != "P7":
        raise Error("read_pam: not a PAM (P7) file: " + path)
    var width = -1
    var height = -1
    var depth = -1
    var maxval = -1
    while True:
        var tok = _read_token(bytes, pos)
        if tok == "ENDHDR":
            break
        elif tok == "WIDTH":
            width = _read_int(bytes, pos)
        elif tok == "HEIGHT":
            height = _read_int(bytes, pos)
        elif tok == "DEPTH":
            depth = _read_int(bytes, pos)
        elif tok == "MAXVAL":
            maxval = _read_int(bytes, pos)
        elif tok == "TUPLTYPE":
            _ = _read_token(bytes, pos)  # RGB / RGB_ALPHA — depth is authoritative
        else:
            raise Error("read_pam: unknown header token " + tok)
    # exactly one whitespace byte (newline) after ENDHDR, then the raster
    pos += 1
    if width <= 0 or height <= 0:
        raise Error("read_pam: missing WIDTH/HEIGHT")
    if depth != 3 and depth != 4:
        raise Error("read_pam: only DEPTH 3/4 supported, got " + String(depth))
    if maxval != 255:
        raise Error("read_pam: only MAXVAL 255 supported")
    var img = ImageU8(width, height, depth)
    _copy_raster(bytes, pos, img)
    return img^


def read_ppm(path: String) raises -> ImageU8:
    """PPM P6 (RGB), maxval 255."""
    var bytes = read_file_bytes(path)
    var pos = 0
    if _read_token(bytes, pos) != "P6":
        raise Error("read_ppm: not a binary PPM (P6) file: " + path)
    var width = _read_int(bytes, pos)
    var height = _read_int(bytes, pos)
    var maxval = _read_int(bytes, pos)
    if maxval != 255:
        raise Error("read_ppm: only MAXVAL 255 supported")
    pos += 1  # single whitespace byte after maxval, then the raster
    var img = ImageU8(width, height, 3)
    _copy_raster(bytes, pos, img)
    return img^


def read_image(path: String) raises -> ImageU8:
    """Dispatch on the magic bytes: P7 -> PAM, P6 -> PPM."""
    var bytes = read_file_bytes(path)
    if len(bytes) >= 2 and bytes[0] == 0x50 and bytes[1] == 0x37:
        return read_pam(path)
    if len(bytes) >= 2 and bytes[0] == 0x50 and bytes[1] == 0x36:
        return read_ppm(path)
    raise Error(
        "read_image: unsupported format (need PAM P7 or PPM P6): " + path
        + " — convert with the PIL one-liner in README_MOJO.md"
    )
