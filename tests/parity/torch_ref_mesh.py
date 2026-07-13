"""Reference side of the mesh-extraction parity test (WP9 part 3 step 4).

Runs the vendored o_voxel stub (tests/parity/o_voxel_stub.py — identical
output to the CUDA version for inference) on seeded dual-grid cases shaped
like fdg_head output: a dense voxel block (many complete quads) plus
scattered voxels (quads rejected on missing neighbors), dual vertices in
[-margin, 1+margin], random intersection flags, softplus split weights.
Cases cover the pipeline path (split_weight) and the normal-alignment
branch (split_weight=None), plus empty/no-valid-quad degenerates.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tests.parity import o_voxel_stub  # noqa: E402

AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


def _case_inputs(seed, grid, block, scattered, flag_density):
    """Dense block^3 of voxels at a seeded offset + scattered singles."""
    g = torch.Generator().manual_seed(seed + 4000)
    off = torch.randint(0, grid - block, (3,), generator=g)
    zz, yy, xx = torch.meshgrid(
        torch.arange(block), torch.arange(block), torch.arange(block), indexing="ij"
    )
    dense = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3) + off
    extra = torch.randint(0, grid, (scattered, 3), generator=g)
    coords = torch.cat([dense, extra], dim=0)
    coords = torch.unique(coords, dim=0)  # sorted + deduped, deterministic
    n = coords.shape[0]
    dual = torch.rand(n, 3, generator=g) * 2 - 0.5
    flags = (torch.rand(n, 3, generator=g) < flag_density)
    sw = F.softplus(torch.randn(n, 1, generator=g))
    return coords.int(), dual, flags, sw


def ref_mesh(seed, grid=12, block=4, scattered=30, flag_density=0.45, use_split_weight=True):
    coords, dual, flags, sw = _case_inputs(seed, grid, block, scattered, flag_density)
    v, t = o_voxel_stub.flexible_dual_grid_to_mesh(
        coords, dual, flags, sw if use_split_weight else None, aabb=AABB, grid_size=grid
    )
    return {
        "coords": coords, "dual": dual, "flags": flags.float(), "sw": sw,
        "vertices": v, "triangles": t, "grid": grid,
    }


def ref_mesh_degenerate(which):
    """which='noflags': no intersected edges; which='noquads': flags set
    but every voxel isolated -> no complete quads."""
    g = torch.Generator().manual_seed(4100)
    coords = (torch.arange(5, dtype=torch.int32).reshape(-1, 1) * 3).expand(-1, 3).contiguous()
    n = coords.shape[0]
    dual = torch.rand(n, 3, generator=g)
    flags = torch.zeros(n, 3, dtype=torch.bool) if which == "noflags" else torch.ones(n, 3, dtype=torch.bool)
    sw = F.softplus(torch.randn(n, 1, generator=g))
    v, t = o_voxel_stub.flexible_dual_grid_to_mesh(
        coords, dual, flags, sw, aabb=AABB, grid_size=16
    )
    return {
        "coords": coords, "dual": dual, "flags": flags.float(), "sw": sw,
        "vertices": v, "triangles": t, "grid": 16,
    }


def check_obj(path, vertices, triangles):
    """Re-read an OBJ written by the Mojo writer and verify it matches."""
    vs, fs = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vs.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                fs.append([int(x) - 1 for x in parts[1:4]])
    vs = torch.tensor(vs, dtype=torch.float32).reshape(-1, 3)
    fs = torch.tensor(fs, dtype=torch.long).reshape(-1, 3)
    assert vs.shape == vertices.shape, f"obj vertex count {vs.shape} != {vertices.shape}"
    assert fs.shape == triangles.shape, f"obj face count {fs.shape} != {triangles.shape}"
    assert torch.equal(fs, triangles.cpu()), "obj faces differ"
    dv = (vs - vertices.cpu()).abs().max().item()
    assert dv < 1e-5, f"obj vertices differ by {dv}"
    return True
