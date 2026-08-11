# Small pure-Mojo CLI number parsers. Keeping these local avoids importing
# CPython just to parse --seed/--steps and remesh tuning values.

comptime MINUS = UInt8(45)
comptime PLUS = UInt8(43)
comptime DOT = UInt8(46)


def parse_int(s: String) raises -> Int:
    var b = s.as_bytes()
    var n = len(b)
    var i = 0
    var neg = False
    if n > 0 and (b[0] == MINUS or b[0] == PLUS):
        neg = b[0] == MINUS
        i = 1
    if i >= n:
        raise Error("invalid integer: " + s)
    var value = 0
    while i < n:
        var c = b[i]
        if c < 48 or c > 57:
            raise Error("invalid integer: " + s)
        value = value * 10 + Int(c - 48)
        i += 1
    return -value if neg else value


def parse_float(s: String) raises -> Float64:
    var b = s.as_bytes()
    var n = len(b)
    var i = 0
    var neg = False
    if n > 0 and (b[0] == MINUS or b[0] == PLUS):
        neg = b[0] == MINUS
        i = 1

    var value: Float64 = 0.0
    var seen = False
    while i < n and b[i] >= 48 and b[i] <= 57:
        value = value * 10.0 + Float64(Int(b[i] - 48))
        i += 1
        seen = True
    if i < n and b[i] == DOT:
        i += 1
        var scale: Float64 = 0.1
        while i < n and b[i] >= 48 and b[i] <= 57:
            value += Float64(Int(b[i] - 48)) * scale
            scale *= 0.1
            i += 1
            seen = True
    if not seen:
        raise Error("invalid number: " + s)

    if i < n and (b[i] == 101 or b[i] == 69):  # e/E
        i += 1
        var exp_neg = False
        if i < n and (b[i] == MINUS or b[i] == PLUS):
            exp_neg = b[i] == MINUS
            i += 1
        var exponent = 0
        var exp_seen = False
        while i < n and b[i] >= 48 and b[i] <= 57:
            exponent = exponent * 10 + Int(b[i] - 48)
            i += 1
            exp_seen = True
        if not exp_seen:
            raise Error("invalid exponent: " + s)
        var power: Float64 = 1.0
        for _ in range(exponent):
            power *= 10.0
        value = value / power if exp_neg else value * power

    if i != n:
        raise Error("invalid number: " + s)
    return -value if neg else value
