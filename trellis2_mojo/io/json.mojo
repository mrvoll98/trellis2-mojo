# Mini JSON parser in pure Mojo (WP12). Covers exactly what the runner
# path needs: safetensors headers, the ckpt config JSONs and pipeline.json
# — objects, arrays, strings (with escapes incl. \uXXXX + surrogate
# pairs), numbers, booleans, null.
#
# Arena representation instead of a recursive value type: JsonDoc owns
# parallel per-node lists and hands out node INDICES. Object key lookup is
# a linear scan — the largest object we parse is a safetensors header
# (~700 keys), where callers iterate obj_keys() anyway.
#
# Number precision: mantissa accumulates in Int (exact to 18 digits) and
# is scaled by an exact power of ten in one f64 multiply/divide, so the
# short decimals in the configs (<= 7 significant digits) parse to the
# SAME f64 as Python/strtod. Mantissas beyond 18 digits lose their tail —
# none of our inputs have them (verified by the io parity test).

comptime J_NULL = 0
comptime J_BOOL = 1
comptime J_NUM = 2
comptime J_STR = 3
comptime J_ARR = 4
comptime J_OBJ = 5


struct JsonDoc(Movable):
    var kind: List[Int]
    var num: List[Float64]       # J_NUM value; J_BOOL stores 0/1 here
    var text: List[String]       # J_STR value
    var kids: List[List[Int]]    # J_ARR/J_OBJ children (node indices)
    var keys: List[List[String]] # J_OBJ keys, parallel to kids
    var root: Int

    def __init__(out self):
        self.kind = List[Int]()
        self.num = List[Float64]()
        self.text = List[String]()
        self.kids = List[List[Int]]()
        self.keys = List[List[String]]()
        self.root = -1

    def _new_node(mut self, k: Int) -> Int:
        self.kind.append(k)
        self.num.append(0.0)
        self.text.append(String(""))
        self.kids.append(List[Int]())
        self.keys.append(List[String]())
        return len(self.kind) - 1

    # -- accessors ---------------------------------------------------------

    def obj_has(self, node: Int, key: String) raises -> Bool:
        if self.kind[node] != J_OBJ:
            raise Error("json: not an object")
        for i in range(len(self.keys[node])):
            if self.keys[node][i] == key:
                return True
        return False

    def obj_get(self, node: Int, key: String) raises -> Int:
        if self.kind[node] != J_OBJ:
            raise Error("json: not an object")
        for i in range(len(self.keys[node])):
            if self.keys[node][i] == key:
                return self.kids[node][i]
        raise Error("json: key not found: " + key)

    def obj_keys(self, node: Int) raises -> List[String]:
        if self.kind[node] != J_OBJ:
            raise Error("json: not an object")
        return self.keys[node].copy()

    def arr_len(self, node: Int) raises -> Int:
        if self.kind[node] != J_ARR and self.kind[node] != J_OBJ:
            raise Error("json: not an array")
        return len(self.kids[node])

    def arr_at(self, node: Int, i: Int) raises -> Int:
        if self.kind[node] != J_ARR:
            raise Error("json: not an array")
        return self.kids[node][i]

    def get_str(self, node: Int) raises -> String:
        if self.kind[node] != J_STR:
            raise Error("json: not a string")
        return self.text[node].copy()

    def get_float(self, node: Int) raises -> Float64:
        if self.kind[node] != J_NUM:
            raise Error("json: not a number")
        return self.num[node]

    def get_int(self, node: Int) raises -> Int:
        if self.kind[node] != J_NUM:
            raise Error("json: not a number")
        return Int(self.num[node])

    def get_bool(self, node: Int) raises -> Bool:
        if self.kind[node] != J_BOOL:
            raise Error("json: not a bool")
        return self.num[node] != 0.0

    def is_null(self, node: Int) -> Bool:
        return self.kind[node] == J_NULL


def _skip_ws(b: List[UInt8], mut pos: Int):
    while pos < len(b):
        var c = b[pos]
        if c == 32 or c == 9 or c == 10 or c == 13:
            pos += 1
        else:
            break


def _expect(b: List[UInt8], mut pos: Int, c: UInt8) raises:
    if pos >= len(b) or b[pos] != c:
        raise Error("json: expected '" + String(Int(c)) + "' at byte " + String(pos))
    pos += 1


def _hex4(b: List[UInt8], mut pos: Int) raises -> Int:
    var v = 0
    for _ in range(4):
        if pos >= len(b):
            raise Error("json: truncated \\u escape")
        var c = Int(b[pos])
        var d: Int
        if c >= 48 and c <= 57:
            d = c - 48
        elif c >= 97 and c <= 102:
            d = c - 87
        elif c >= 65 and c <= 70:
            d = c - 55
        else:
            raise Error("json: bad hex digit in \\u escape")
        v = v * 16 + d
        pos += 1
    return v


def _push_utf8(mut out: List[UInt8], cp: Int):
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


def _parse_string(b: List[UInt8], mut pos: Int) raises -> String:
    _expect(b, pos, 34)  # '"'
    var out = List[UInt8]()
    while True:
        if pos >= len(b):
            raise Error("json: unterminated string")
        var c = b[pos]
        pos += 1
        if c == 34:
            break
        if c != 92:  # not '\'
            out.append(c)
            continue
        if pos >= len(b):
            raise Error("json: dangling escape")
        var e = b[pos]
        pos += 1
        if e == 34 or e == 92 or e == 47:  # " \ /
            out.append(e)
        elif e == 98:
            out.append(8)   # \b
        elif e == 102:
            out.append(12)  # \f
        elif e == 110:
            out.append(10)  # \n
        elif e == 114:
            out.append(13)  # \r
        elif e == 116:
            out.append(9)   # \t
        elif e == 117:  # \uXXXX
            var cp = _hex4(b, pos)
            if cp >= 0xD800 and cp <= 0xDBFF:
                # surrogate pair: require the low half right after
                if pos + 1 >= len(b) or b[pos] != 92 or b[pos + 1] != 117:
                    raise Error("json: lone high surrogate")
                pos += 2
                var lo = _hex4(b, pos)
                if lo < 0xDC00 or lo > 0xDFFF:
                    raise Error("json: bad low surrogate")
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
            elif cp >= 0xDC00 and cp <= 0xDFFF:
                raise Error("json: lone low surrogate")
            _push_utf8(out, cp)
        else:
            raise Error("json: bad escape")
    return String(from_utf8=Span(out))


def _pow10(k: Int) -> Float64:
    # 10^k is exactly representable in f64 for k <= 22, and the running
    # product stays exact on the way up.
    var f: Float64 = 1.0
    for _ in range(k):
        f *= 10.0
    return f


def _parse_number(b: List[UInt8], mut pos: Int) raises -> Float64:
    var neg = False
    if pos < len(b) and b[pos] == 45:  # '-'
        neg = True
        pos += 1
    var mant = 0
    var ndig = 0
    var exp10 = 0
    var any_digit = False
    while pos < len(b) and b[pos] >= 48 and b[pos] <= 57:
        any_digit = True
        if ndig < 18:
            mant = mant * 10 + Int(b[pos] - 48)
            ndig += 1
        else:
            exp10 += 1  # drop tail digits, keep magnitude
        pos += 1
    if pos < len(b) and b[pos] == 46:  # '.'
        pos += 1
        while pos < len(b) and b[pos] >= 48 and b[pos] <= 57:
            any_digit = True
            if ndig < 18:
                mant = mant * 10 + Int(b[pos] - 48)
                ndig += 1
                exp10 -= 1
            pos += 1
    if not any_digit:
        raise Error("json: bad number at byte " + String(pos))
    if pos < len(b) and (b[pos] == 101 or b[pos] == 69):  # e/E
        pos += 1
        var eneg = False
        if pos < len(b) and (b[pos] == 43 or b[pos] == 45):
            eneg = b[pos] == 45
            pos += 1
        var e = 0
        var any_e = False
        while pos < len(b) and b[pos] >= 48 and b[pos] <= 57:
            any_e = True
            e = e * 10 + Int(b[pos] - 48)
            pos += 1
        if not any_e:
            raise Error("json: bad exponent")
        exp10 += -e if eneg else e
    var f = Float64(mant)
    if exp10 > 0:
        if exp10 <= 22:
            f *= _pow10(exp10)
        else:
            f *= _pow10(22)
            f *= _pow10(exp10 - 22)
    elif exp10 < 0:
        if -exp10 <= 22:
            f /= _pow10(-exp10)
        else:
            f /= _pow10(22)
            f /= _pow10(-exp10 - 22)
    return -f if neg else f


def _parse_literal(b: List[UInt8], mut pos: Int, lit: String) raises:
    var lb = lit.as_bytes()
    for i in range(len(lb)):
        if pos + i >= len(b) or b[pos + i] != lb[i]:
            raise Error("json: bad literal at byte " + String(pos))
    pos += len(lb)


def _parse_value(mut doc: JsonDoc, b: List[UInt8], mut pos: Int) raises -> Int:
    _skip_ws(b, pos)
    if pos >= len(b):
        raise Error("json: unexpected end of input")
    var c = b[pos]
    if c == 123:  # '{'
        pos += 1
        var node = doc._new_node(J_OBJ)
        _skip_ws(b, pos)
        if pos < len(b) and b[pos] == 125:  # '}'
            pos += 1
            return node
        while True:
            _skip_ws(b, pos)
            var key = _parse_string(b, pos)
            _skip_ws(b, pos)
            _expect(b, pos, 58)  # ':'
            var child = _parse_value(doc, b, pos)
            doc.keys[node].append(key)
            doc.kids[node].append(child)
            _skip_ws(b, pos)
            if pos >= len(b):
                raise Error("json: unterminated object")
            if b[pos] == 44:  # ','
                pos += 1
                continue
            _expect(b, pos, 125)  # '}'
            return node
    if c == 91:  # '['
        pos += 1
        var node = doc._new_node(J_ARR)
        _skip_ws(b, pos)
        if pos < len(b) and b[pos] == 93:  # ']'
            pos += 1
            return node
        while True:
            var child = _parse_value(doc, b, pos)
            doc.kids[node].append(child)
            _skip_ws(b, pos)
            if pos >= len(b):
                raise Error("json: unterminated array")
            if b[pos] == 44:  # ','
                pos += 1
                continue
            _expect(b, pos, 93)  # ']'
            return node
    if c == 34:  # '"'
        var s = _parse_string(b, pos)
        var node = doc._new_node(J_STR)
        doc.text[node] = s^
        return node
    if c == 116:  # true
        _parse_literal(b, pos, "true")
        var node = doc._new_node(J_BOOL)
        doc.num[node] = 1.0
        return node
    if c == 102:  # false
        _parse_literal(b, pos, "false")
        return doc._new_node(J_BOOL)
    if c == 110:  # null
        _parse_literal(b, pos, "null")
        return doc._new_node(J_NULL)
    var v = _parse_number(b, pos)
    var node = doc._new_node(J_NUM)
    doc.num[node] = v
    return node


def parse_json(b: List[UInt8]) raises -> JsonDoc:
    """Parse a complete JSON document; trailing whitespace is allowed
    (safetensors pads headers with spaces)."""
    var doc = JsonDoc()
    var pos = 0
    doc.root = _parse_value(doc, b, pos)
    _skip_ws(b, pos)
    if pos != len(b):
        raise Error("json: trailing garbage at byte " + String(pos))
    return doc^


def parse_json_str(s: String) raises -> JsonDoc:
    var b = List[UInt8]()
    for c in s.as_bytes():
        b.append(c)
    return parse_json(b)
