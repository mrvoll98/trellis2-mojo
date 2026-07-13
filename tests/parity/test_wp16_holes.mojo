# WP16 (fase 2): small-hole filling — pure-Mojo tests (cumesh, the
# upstream reference, is a closed CUDA library, so this ports the intent,
# not bits; see the ADR 0008 addendum). Covers: a punctured tetrahedron
# becomes watertight with consistent winding, the perimeter threshold is
# respected, an open patch's real opening is left alone, two micro holes
# sharing a boundary vertex both close (the commit/revert walk), the fill
# is deterministic, and watertight input is untouched.
#
# Run from repo root: pixi run test-wp16

from std.collections import Dict

from trellis2_mojo.meshing.postprocess import (
    fill_small_holes,
    remove_small_connected_components,
    repair_non_manifold_edges,
    sew_boundary_seams,
    unify_face_orientations,
)
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32


def make_faces(rows: List[Int]) raises -> IntMatrix:
    var m = IntMatrix(len(rows) // 3, 3)
    for i in range(len(rows)):
        m.data[i] = Int32(rows[i])
    return m^


def edge_counts(faces: IntMatrix, v: Int) raises -> Dict[Int, Int]:
    var cnt = Dict[Int, Int]()
    for f in range(faces.rows):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var key: Int
            if a < b:
                key = a * v + b
            else:
                key = b * v + a
            if key in cnt:
                cnt[key] += 1
            else:
                cnt[key] = 1
    return cnt^


def boundary_edges(faces: IntMatrix, v: Int) raises -> Int:
    var cnt = edge_counts(faces, v)
    var n = 0
    for e in cnt.items():
        if e.value == 1:
            n += 1
    return n


def check_consistent_winding(faces: IntMatrix, v: Int) raises:
    """Every directed edge must appear exactly once (its reverse belongs
    to the neighboring triangle) — the manifold-orientation invariant."""
    var d = Dict[Int, Int]()
    for f in range(faces.rows):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var key = a * v + b
            if key in d:
                raise Error("directed edge repeated -> inconsistent winding")
            d[key] = 1


def tetra() raises -> Tuple[Tensor[F32], IntMatrix]:
    """Unit-ish tetrahedron with one face REMOVED (outward winding)."""
    var verts = Tensor[F32]([4, 3])
    var vals: List[Float32] = [
        0.0, 0.0, 0.0,
        0.005, 0.0, 0.0,
        0.0, 0.005, 0.0,
        0.0, 0.0, 0.005,
    ]
    for i in range(12):
        verts.data[i] = vals[i]
    # full tetra: (0,2,1) (0,1,3) (1,2,3) (0,3,2); drop (0,2,1)
    var rows: List[Int] = [0, 1, 3, 1, 2, 3, 0, 3, 2]
    return (verts^, make_faces(rows))


def grid_patch(
    w: Int, h: Int, s: Float32, skip1: Int, skip2: Int
) raises -> Tuple[Tensor[F32], IntMatrix]:
    """Open (w-1)x(h-1)-quad plane with spacing s; quads skip1/skip2
    (linear quad index, -1 = none) are left out."""
    var verts = Tensor[F32]([w * h, 3])
    for i in range(w):
        for j in range(h):
            verts.data[(i * h + j) * 3 + 0] = Float32(i) * s
            verts.data[(i * h + j) * 3 + 1] = Float32(j) * s
            verts.data[(i * h + j) * 3 + 2] = 0
    var rows = List[Int]()
    for i in range(w - 1):
        for j in range(h - 1):
            var q = i * (h - 1) + j
            if q == skip1 or q == skip2:
                continue
            var a = i * h + j
            var b = (i + 1) * h + j
            var c = (i + 1) * h + j + 1
            var d = i * h + j + 1
            rows.append(a)
            rows.append(b)
            rows.append(c)
            rows.append(a)
            rows.append(c)
            rows.append(d)
    return (verts^, make_faces(rows))


def main() raises:
    # 1) punctured tetrahedron -> watertight, consistent winding
    var t = tetra()
    var tv = t[0].copy()
    var tf = IntMatrix(t[1].rows, 3)
    for i in range(len(t[1].data)):
        tf.data[i] = t[1].data[i]
    var r = fill_small_holes(tv, tf, 3e-2)
    if r[0] != 1 or r[1] != 3:
        raise Error("tetra: expected 1 hole from 3 boundary edges")
    if tv.shape[0] != 5 or tf.rows != 6:
        raise Error("tetra: expected 1 new vertex and 3 new faces")
    if boundary_edges(tf, tv.shape[0]) != 0:
        raise Error("tetra: still has boundary edges")
    check_consistent_winding(tf, tv.shape[0])
    print("  tetra: hole filled, watertight, winding consistent")

    # 2) threshold respected: same hole, tiny threshold -> untouched
    var t2 = tetra()
    var tv2 = t2[0].copy()
    var tf2 = IntMatrix(t2[1].rows, 3)
    for i in range(len(t2[1].data)):
        tf2.data[i] = t2[1].data[i]
    var r2 = fill_small_holes(tv2, tf2, 1e-4)
    if r2[0] != 0 or tv2.shape[0] != 4 or tf2.rows != 3:
        raise Error("tetra: tiny threshold must fill nothing")
    print("  threshold: sub-perimeter hole left alone")

    # 3) open patch: the outer rim (16 edges x 0.001 = 0.016) is a real
    # opening at threshold 5e-3 -> untouched; at 3e-2 it gets closed
    var p = grid_patch(5, 5, 0.001, -1, -1)
    var pv = p[0].copy()
    var pf = IntMatrix(p[1].rows, 3)
    for i in range(len(p[1].data)):
        pf.data[i] = p[1].data[i]
    var rp = fill_small_holes(pv, pf, 5e-3)
    if rp[0] != 0 or rp[1] != 16:
        raise Error("patch: rim must stay open below threshold")
    print("  patch rim (perimeter 0.016) left open at 5e-3")

    # 4) two micro holes sharing a boundary vertex: union-find merges
    # them into ONE component (cumesh semantics — braids and junctions
    # fill as one centroid fan); the combined perimeter 0.008 must still
    # clear the 5e-3-per-hole intuition, so threshold 1e-2 here
    var g = grid_patch(5, 5, 0.001, 5, 10)
    var gv = g[0].copy()
    var gf = IntMatrix(g[1].rows, 3)
    for i in range(len(g[1].data)):
        gf.data[i] = g[1].data[i]
    var before = boundary_edges(gf, gv.shape[0])
    var rg = fill_small_holes(gv, gf, 1e-2)
    if rg[0] != 1:
        raise Error(
            "shared-vertex holes: expected 1 merged component, got "
            + String(rg[0])
        )
    var after = boundary_edges(gf, gv.shape[0])
    if before - after != 8:
        raise Error("shared-vertex holes: 8 boundary edges must close")
    print("  two holes sharing a vertex: filled as one component (", before, "->", after, "boundary edges )")

    # 5) determinism: identical reruns give identical buffers
    var g2 = grid_patch(5, 5, 0.001, 5, 10)
    var gv2 = g2[0].copy()
    var gf2 = IntMatrix(g2[1].rows, 3)
    for i in range(len(g2[1].data)):
        gf2.data[i] = g2[1].data[i]
    _ = fill_small_holes(gv2, gf2, 1e-2)
    if len(gv2.data) != len(gv.data) or gf2.rows != gf.rows:
        raise Error("determinism: sizes differ")
    for i in range(len(gv.data)):
        if gv.data[i] != gv2.data[i]:
            raise Error("determinism: vertex data differs")
    for i in range(len(gf.data)):
        if gf.data[i] != gf2.data[i]:
            raise Error("determinism: face data differs")
    print("  deterministic across reruns")

    # 6) watertight input untouched (the filled tetra from case 1)
    var r6 = fill_small_holes(tv, tf, 3e-2)
    if r6[0] != 0 or r6[1] != 0:
        raise Error("watertight mesh must report 0 boundary edges")
    print("  watertight input untouched")

    # 7) a hole ring where one edge is NON-MANIFOLD (two fins attached ->
    # count 3): the ring reads as a dead-end path and fill alone skips it
    # (this is the user-visible leftover class — 100% of the golden's
    # dead ends sat on non-manifold edges). repair_non_manifold_edges
    # splits the fans into manifold sheets, after which every boundary
    # is a closed loop and the second fill closes everything.
    var t7 = tetra()
    var tv7 = t7[0].copy()
    # tetra minus one face + two fins on hole-ring edge (0, 1)
    var extra: List[Float32] = [0.002, 0.002, 0.005, 0.002, 0.002, -0.005]
    for x in extra:
        tv7.data.append(x)
    tv7.shape[0] = 6
    var rows7: List[Int] = [
        0, 1, 3, 1, 2, 3, 0, 3, 2,  # punctured tetra
        0, 1, 4, 1, 0, 5,           # two fins -> edge (0,1) has count 3
    ]
    var tf7 = make_faces(rows7)
    var r7a = fill_small_holes(tv7, tf7, 3e-2)
    if r7a[0] != 0:
        raise Error("non-manifold ring: fill alone must not close anything")
    var nv7 = repair_non_manifold_edges(tv7, tf7)
    var r7b = fill_small_holes(tv7, tf7, 3e-2)
    if r7b[0] < 1:
        raise Error("non-manifold ring: repair + fill must close the ring")
    if boundary_edges(tf7, tv7.shape[0]) != 0:
        raise Error("non-manifold ring: boundary edges remain after repair+fill")
    print(
        "  non-manifold ring: fill 0 -> repair (", nv7,
        "verts ) -> fill", r7b[0], "components, boundary 0",
    )

    # 8) orientation unify (parity-only, cumesh semantics): a closed cube
    # with HALF the faces deliberately flipped — after unify every
    # manifold pair must be opposite-direction (the directed-edge-unique
    # invariant) and the sheet count must be 1. No global in/out vote
    # (measured useless on the folded FDG surface — see postprocess.mojo).
    var cw = 8
    var cverts = Tensor[F32]([cw, 3])
    var k = 0
    for z in range(2):
        for y in range(2):
            for x in range(2):
                cverts.data[k * 3 + 0] = Float32(x) * 0.5 - 0.25
                cverts.data[k * 3 + 1] = Float32(y) * 0.5 - 0.25
                cverts.data[k * 3 + 2] = Float32(z) * 0.5 - 0.25
                k += 1
    # outward-wound cube (vertex bits: x + 2y + 4z), then flip 6 of 12
    var crows: List[Int] = [
        0, 2, 1, 1, 2, 3,  # z = -0.25
        4, 5, 6, 5, 7, 6,  # z = +0.25
        0, 1, 4, 1, 5, 4,  # y = -0.25
        2, 6, 3, 3, 6, 7,  # y = +0.25
        0, 4, 2, 2, 4, 6,  # x = -0.25
        1, 3, 5, 3, 7, 5,  # x = +0.25
    ]
    var cfaces = make_faces(crows)
    for f in range(0, 12, 2):  # flip every other face
        var tmp = cfaces.data[f * 3 + 1]
        cfaces.data[f * 3 + 1] = cfaces.data[f * 3 + 2]
        cfaces.data[f * 3 + 2] = tmp
    var u8 = unify_face_orientations(cfaces, cverts)
    check_consistent_winding(cfaces, cw)
    if u8[1] != 1:
        raise Error("cube must be one sheet")
    print("  orientation unify: mixed cube -> consistent winding (", u8[0], "flipped )")
    # deterministic across reruns on identical input
    var cfaces2 = make_faces(crows)
    for f in range(0, 12, 2):
        var tmp = cfaces2.data[f * 3 + 1]
        cfaces2.data[f * 3 + 1] = cfaces2.data[f * 3 + 2]
        cfaces2.data[f * 3 + 2] = tmp
    _ = unify_face_orientations(cfaces2, cverts)
    for i in range(len(cfaces.data)):
        if cfaces.data[i] != cfaces2.data[i]:
            raise Error("orientation unify must be deterministic")
    print("  orientation unify: deterministic")

    # 9) remove_small_connected_components (cumesh port): components are
    # unioned ONLY across manifold edges (exactly 2 incident faces) and
    # judged on their area SUM. Scene: a 5x5 grid patch (32 faces, total
    # area 1.6e-5), a big isolated triangle (5e-3), a tiny isolated
    # triangle (5e-7), a tiny triangle SHARING A VERTEX with the grid
    # (a vertex does not connect -> separate component, removed), and a
    # tiny EDGE-sharing triangle PAIR (one merged component of 1e-6).
    var s9 = grid_patch(5, 5, 0.001, -1, -1)
    var sv = s9[0].copy()  # 25 grid verts
    var extra9: List[Float32] = [
        # big isolated triangle: legs 0.1 -> area 5e-3 (verts 25-27)
        0.2, 0.2, 0.0, 0.3, 0.2, 0.0, 0.2, 0.3, 0.0,
        # tiny isolated triangle: legs 0.001 -> area 5e-7 (verts 28-30)
        0.5, 0.5, 0.0, 0.501, 0.5, 0.0, 0.5, 0.501, 0.0,
        # tiny triangle sharing grid vertex 0 (verts 31-32)
        -0.001, 0.0, 0.0, 0.0, -0.001, 0.0,
        # tiny edge-sharing pair: quad diagonal (verts 33-36)
        0.7, 0.7, 0.0, 0.701, 0.7, 0.0, 0.701, 0.701, 0.0, 0.7, 0.701, 0.0,
    ]
    for x in extra9:
        sv.data.append(x)
    sv.shape[0] = 37
    var sf = IntMatrix(s9[1].rows, 3)
    for i in range(len(s9[1].data)):
        sf.data[i] = s9[1].data[i]
    var extra_faces: List[Int] = [
        25, 26, 27,  # big isolated
        28, 29, 30,  # tiny isolated
        0, 31, 32,   # tiny, shares grid vertex 0 only
        33, 34, 35, 33, 35, 36,  # tiny pair across a manifold edge
    ]
    for i in range(0, len(extra_faces), 3):
        sf.data.append(Int32(extra_faces[i]))
        sf.data.append(Int32(extra_faces[i + 1]))
        sf.data.append(Int32(extra_faces[i + 2]))
        sf.rows += 1
    var nf9 = sf.rows
    var r9 = remove_small_connected_components(sv, sf, 1e-5)
    # removed: tiny isolated + vertex-sharing tiny + edge-sharing pair
    # (ONE merged component) = 3 components, 4 faces; the grid (32 tiny
    # faces SUMMING to 1.6e-5) and the big triangle survive
    if r9[0] != 3 or r9[1] != 4:
        raise Error(
            "small components: expected 3 components / 4 faces removed, got "
            + String(r9[0]) + " / " + String(r9[1])
        )
    if sf.rows != nf9 - 4 or sv.shape[0] != 28:
        raise Error("small components: survivors/vertices mismatch")
    # vertex compaction keeps original order: grid verts 0-24 unchanged,
    # big triangle now verts 25-27 with the same positions
    if sv.data[0] != 0.0 or sv.data[25 * 3 + 0] != Float32(0.2):
        raise Error("small components: vertex remap broke positions")
    var last9 = sf.rows - 1
    if sf.at(last9, 0) != 25 or sf.at(last9, 1) != 26 or sf.at(last9, 2) != 27:
        raise Error("small components: face remap broke indices")
    check_consistent_winding(sf, sv.shape[0])
    print(
        "  small components: 3 removed (vertex-share + merged edge-pair),",
        "grid area-sum + big triangle kept, verts 37 ->", sv.shape[0],
    )
    # determinism + no-op on the cleaned mesh
    var r9b = remove_small_connected_components(sv, sf, 1e-5)
    if r9b[0] != 0 or r9b[1] != 0:
        raise Error("small components: second pass must be a no-op")
    print("  small components: idempotent on cleaned mesh")

    # 10) sew_boundary_seams (WP16 v7, own semantics): two open quads
    # whose shared border is DUPLICATED with bit-identical positions
    # (the crack class) weld into one patch; 1-ulp-offset twins do NOT
    # weld (no epsilon); a sliver spanning the seam degenerates and is
    # dropped; deterministic; watertight input untouched.
    def two_quads(offset: Float32) raises -> Tuple[Tensor[F32], IntMatrix]:
        """Quad A (verts 0-3) + quad B (verts 4-7); B's verts 4/7 copy
        A's 1/2 positions shifted by `offset` in x (0 = exact twins)."""
        var qv = Tensor[F32]([8, 3])
        var s: Float32 = 0.001
        var vals: List[Float32] = [
            0.0, 0.0, 0.0,   s, 0.0, 0.0,   s, s, 0.0,   0.0, s, 0.0,
            s + offset, 0.0, 0.0,   s + s, 0.0, 0.0,
            s + s, s, 0.0,   s + offset, s, 0.0,
        ]
        for i in range(24):
            qv.data[i] = vals[i]
        var qrows: List[Int] = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
        return (qv^, make_faces(qrows))

    var q10 = two_quads(0.0)
    var qv10 = q10[0].copy()
    var qf10 = IntMatrix(q10[1].rows, 3)
    for i in range(len(q10[1].data)):
        qf10.data[i] = q10[1].data[i]
    var before10 = boundary_edges(qf10, qv10.shape[0])
    var s10 = sew_boundary_seams(qv10, qf10)
    if s10[0] != 2 or s10[1] != 0:
        raise Error(
            "sew: expected 2 welded verts / 0 dropped, got "
            + String(s10[0]) + " / " + String(s10[1])
        )
    if qv10.shape[0] != 6 or qf10.rows != 4:
        raise Error("sew: expected 6 verts / 4 faces after weld")
    var after10 = boundary_edges(qf10, qv10.shape[0])
    if before10 != 8 or after10 != 6:
        raise Error("sew: boundary must go 8 -> 6 (seam edge now interior)")
    print("  seam sew: twin border welded ( boundary", before10, "->", after10, ")")

    # 1-ulp-offset twins are real geometry -> untouched
    var q10b = two_quads(1e-7)
    var qv10b = q10b[0].copy()
    var qf10b = IntMatrix(q10b[1].rows, 3)
    for i in range(len(q10b[1].data)):
        qf10b.data[i] = q10b[1].data[i]
    var s10b = sew_boundary_seams(qv10b, qf10b)
    if s10b[0] != 0 or qv10b.shape[0] != 8:
        raise Error("sew: near-coincident verts must NOT weld")
    print("  seam sew: 1e-7-offset twins left alone (no epsilon)")

    # sliver spanning the seam degenerates and is dropped
    var q10c = two_quads(0.0)
    var qv10c = q10c[0].copy()
    var qf10c = IntMatrix(q10c[1].rows, 3)
    for i in range(len(q10c[1].data)):
        qf10c.data[i] = q10c[1].data[i]
    qf10c.data.append(Int32(1))
    qf10c.data.append(Int32(4))
    qf10c.data.append(Int32(5))
    qf10c.rows += 1
    var s10c = sew_boundary_seams(qv10c, qf10c)
    if s10c[1] != 1 or qf10c.rows != 4:
        raise Error("sew: seam-spanning sliver must be dropped")
    print("  seam sew: seam-spanning sliver dropped")

    # determinism
    var q10d = two_quads(0.0)
    var qv10d = q10d[0].copy()
    var qf10d = IntMatrix(q10d[1].rows, 3)
    for i in range(len(q10d[1].data)):
        qf10d.data[i] = q10d[1].data[i]
    _ = sew_boundary_seams(qv10d, qf10d)
    if len(qv10d.data) != len(qv10.data) or qf10d.rows != qf10.rows:
        raise Error("sew determinism: sizes differ")
    for i in range(len(qv10.data)):
        if qv10.data[i] != qv10d.data[i]:
            raise Error("sew determinism: vertex data differs")
    for i in range(len(qf10.data)):
        if qf10.data[i] != qf10d.data[i]:
            raise Error("sew determinism: face data differs")
    print("  seam sew: deterministic")

    # watertight input untouched (the filled tetra from case 1)
    var s10e = sew_boundary_seams(tv, tf)
    if s10e[0] != 0 or s10e[1] != 0:
        raise Error("sew: watertight input must be a no-op")
    print("  seam sew: watertight input untouched")

    print("wp16 hole filling: tetra + threshold + rim + shared-vertex + determinism + watertight + non-manifold ring + orientation + small-components + seam-sew passed")
