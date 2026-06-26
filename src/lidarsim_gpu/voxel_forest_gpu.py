# LiDARSimGPU — derived from slidaRtools (author Nikolai Knapp;
# (C) 2017 Helmholtz-Zentrum fuer Umweltforschung GmbH - UFZ), GPL-3.
# This is a modified, GPU-accelerated reimplementation. Per GPL-3 (5),
# this file is a changed version of the original algorithm. See LICENSE.
"""GPU-accelerated voxel forest construction using PyTorch.

Produces output identical to :func:`make_voxelforest` but replaces the
triple-nested Python for-loop with **fully batched** tensor operations.
All trees of the same crown shape are processed in a single GPU kernel
call — no Python per-tree loop.

Supports ``return_tensors=True`` to keep output as GPU tensors for zero-copy
chaining with :func:`make_lidarscan_gpu` and :func:`make_profile_from_XYZ_gpu`.
"""

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None


# Vectorised geometry tests (operate on 1-D tensors, return boolean mask)

def _in_cylinder_t(x, y, z, xctr, yctr, zbase, radius, height):
    hdist2 = (x - xctr) ** 2 + (y - yctr) ** 2
    return (z >= zbase) & (z <= zbase + height) & (hdist2 <= radius ** 2)

def _in_spheroid_t(x, y, z, xctr, yctr, zctr, radius, height):
    half_h = 0.5 * height
    val = (x - xctr) ** 2 / radius ** 2 + \
          (y - yctr) ** 2 / radius ** 2 + \
          (z - zctr) ** 2 / half_h ** 2
    return val <= 1

def _in_cone_t(x, y, z, xctr, yctr, zbase, radius, height):
    r_at_h = (-radius / height) * (z - zbase) + radius
    hdist2 = (x - xctr) ** 2 + (y - yctr) ** 2
    return (z >= zbase) & (z <= zbase + height) & (hdist2 <= r_at_h ** 2)

def _in_icecone_t(x, y, z, xctr, yctr, zctr, radius, height):
    half_h = 0.5 * height
    lower_base = zctr - half_h
    r_at_h = (radius / half_h) * (z - lower_base)
    hdist2 = (x - xctr) ** 2 + (y - yctr) ** 2
    in_lower = (z >= lower_base) & (z < zctr) & (hdist2 <= r_at_h ** 2)
    sph_val = (x - xctr) ** 2 / radius ** 2 + \
              (y - yctr) ** 2 / radius ** 2 + \
              (z - zctr) ** 2 / half_h ** 2
    in_upper = (z >= zctr) & (sph_val <= 1)
    return in_lower | in_upper

def _in_cone_inverted_t(x, y, z, xctr, yctr, zbase, radius, height):
    """Vectorised inverted cone: apex at zbase, widens upward to zbase+height."""
    r_at_h = (radius / height) * (z - zbase)
    hdist2 = (x - xctr) ** 2 + (y - yctr) ** 2
    return (z >= zbase) & (z <= zbase + height) & (hdist2 <= r_at_h ** 2)

def _in_trumpet_t(x, y, z, xctr, yctr, z0, max_height, max_radius, decay_rate):
    """Vectorised trumpet/root shape: exponential decay from stem base."""
    r = torch.sqrt((x - xctr) ** 2 + (y - yctr) ** 2)
    height_limit = max_height * torch.exp(-decay_rate * r)
    return z <= z0 + height_limit

_CS_DISPATCH = {
    1: (_in_cylinder_t, "zbase"),
    2: (_in_spheroid_t, "zctr"),
    3: (_in_cone_t, "zbase"),
    4: (_in_icecone_t, "zctr"),
    5: (_in_cone_inverted_t, "zbase"),
}


# Batch tree processing — all trees of same CS in one kernel call

def _batch_crown_voxels(sub_df, geom_fn, z_mode, dev, chunk_size=500):
    """Process all trees of one crown shape in batched GPU operations.

    To avoid GPU OOM, trees are processed in chunks of `chunk_size`.
    Within each chunk, ALL candidate voxels are generated and tested
    in a single kernel call (no Python per-tree loop).
    """
    n_trees = len(sub_df)
    xres = torch.tensor(sub_df["Xres"].values, dtype=torch.int64, device=dev)
    yres = torch.tensor(sub_df["Yres"].values, dtype=torch.int64, device=dev)
    crres = torch.tensor(sub_df["CRres"].values, dtype=torch.int64, device=dev)
    spans = torch.tensor(
        np.ceil(sub_df["CDres"].values.astype(np.float64) + 5).astype(np.int64),
        dtype=torch.int64, device=dev,
    )
    heights = torch.tensor(
        np.ceil(sub_df["H"].values + 1).astype(np.int64),
        dtype=torch.int64, device=dev,
    )
    cl_arr = torch.tensor(sub_df["CL"].values, dtype=torch.float64, device=dev)
    h_arr = torch.tensor(sub_df["H"].values, dtype=torch.float64, device=dev)

    result_x, result_y, result_z = [], [], []

    for chunk_start in range(0, n_trees, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_trees)
        c_spans = spans[chunk_start:chunk_end]
        c_heights = heights[chunk_start:chunk_end]
        c_xres = xres[chunk_start:chunk_end]
        c_yres = yres[chunk_start:chunk_end]
        c_crres = crres[chunk_start:chunk_end]
        c_cl = cl_arr[chunk_start:chunk_end]
        c_h = h_arr[chunk_start:chunk_end]
        n_chunk = chunk_end - chunk_start

        # Number of candidates per tree
        n_cands = c_spans * c_spans * c_heights  # shape (n_chunk,)
        total_cands = n_cands.sum().item()

        if total_cands == 0:
            continue

        # Tree index for each candidate
        tree_idx = torch.repeat_interleave(
            torch.arange(n_chunk, device=dev), n_cands
        )

        # Local flat index within each tree's bounding box
        offsets = torch.zeros(n_chunk, dtype=torch.int64, device=dev)
        offsets[1:] = torch.cumsum(n_cands[:-1], dim=0)
        local_idx = torch.arange(total_cands, device=dev) - offsets[tree_idx]

        # Decompose local index into (local_x, local_y, local_z)
        span_t = c_spans[tree_idx]
        area = span_t * span_t

        local_z = local_idx // area + 1
        remainder = local_idx % area
        local_y = remainder // span_t + 1
        local_x = remainder % span_t + 1

        # Convert to voxel coordinates
        x_vox = local_x + c_xres[tree_idx] - c_crres[tree_idx] - 1
        y_vox = local_y + c_yres[tree_idx] - c_crres[tree_idx] - 1
        z_vox = local_z.to(torch.float64)

        # Validity mask: z <= actual tree height
        valid = local_z <= c_heights[tree_idx]

        # Tree geometry parameters (broadcast to all candidates)
        radius = c_crres[tree_idx].to(torch.float64)
        cl = c_cl[tree_idx]
        tree_h = c_h[tree_idx]

        if z_mode == "zbase":
            z_ref = tree_h - cl
        else:
            z_ref = tree_h - cl / 2.0

        xctr = c_xres[tree_idx].to(torch.float64)
        yctr = c_yres[tree_idx].to(torch.float64)

        # Run geometry test on ALL candidates at once
        in_crown = valid & geom_fn(
            x_vox.to(torch.float64),
            y_vox.to(torch.float64),
            z_vox, xctr, yctr, z_ref, radius, cl,
        )

        result_x.append(x_vox[in_crown].to(torch.int64))
        result_y.append(y_vox[in_crown].to(torch.int64))
        result_z.append(z_vox[in_crown].to(torch.int64))

    return result_x, result_y, result_z


def _batch_stem_voxels(sub_df, dev, roots=False, root_ratio=0.15):
    """Generate stem voxels for all trees at once.

    When roots=True, stem starts above the root zone (z >= root_r_int)
    instead of z=1.
    """
    h_arr = sub_df["H"].values
    cl_arr = sub_df["CL"].values
    xres_arr = sub_df["Xres"].values.astype(np.int64)
    yres_arr = sub_df["Yres"].values.astype(np.int64)

    stem_top = np.floor(h_arr - cl_arr).astype(np.int64)

    if roots:
        root_r = h_arr * root_ratio
        stem_bottom = np.ceil(root_r).astype(np.int64)
    else:
        stem_bottom = np.ones(len(h_arr), dtype=np.int64)

    stem_counts = np.clip(stem_top - stem_bottom, 0, None)

    all_x, all_y, all_z = [], [], []
    mask = stem_counts > 0
    if mask.any():
        sc = stem_counts[mask]
        sb = stem_bottom[mask]
        xr = xres_arr[mask]
        yr = yres_arr[mask]

        tree_idx = np.repeat(np.arange(len(sc)), sc)
        z_vals = np.concatenate([np.arange(s, s + c) for s, c in zip(sb, sc)])

        all_x.append(torch.tensor(xr[tree_idx], dtype=torch.int64, device=dev))
        all_y.append(torch.tensor(yr[tree_idx], dtype=torch.int64, device=dev))
        all_z.append(torch.tensor(z_vals, dtype=torch.int64, device=dev))

    return all_x, all_y, all_z


def _batch_root_voxels(trees_df, dev, root_ratio=0.15, decay_rate=0.4,
                       chunk_size=500):
    """Generate trumpet-shaped root voxels for all trees (batched GPU)."""
    n_trees = len(trees_df)
    xres = torch.tensor(trees_df["Xres"].values, dtype=torch.int64, device=dev)
    yres = torch.tensor(trees_df["Yres"].values, dtype=torch.int64, device=dev)
    h_arr = torch.tensor(trees_df["H"].values, dtype=torch.float64, device=dev)

    root_r = h_arr * root_ratio
    root_r_int = torch.ceil(root_r).to(torch.int64)
    above_root_z = torch.ceil(root_r * 1.5).to(torch.int64)

    result_x, result_y, result_z = [], [], []

    for chunk_start in range(0, n_trees, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_trees)
        c_xres = xres[chunk_start:chunk_end]
        c_yres = yres[chunk_start:chunk_end]
        c_root_r_int = root_r_int[chunk_start:chunk_end]
        c_above_root_z = above_root_z[chunk_start:chunk_end]
        n_chunk = chunk_end - chunk_start

        # Each tree's root bounding box: x in [-root_r_int, root_r_int),
        # y in [-root_r_int, root_r_int), z in [0, root_r_int)
        # Span per tree = (2*root_r_int) for x and y, root_r_int for z
        span_xy = 2 * c_root_r_int  # per tree
        span_z = c_root_r_int       # per tree

        n_cands = span_xy * span_xy * span_z
        total_cands = n_cands.sum().item()
        if total_cands == 0:
            continue

        tree_idx = torch.repeat_interleave(
            torch.arange(n_chunk, device=dev), n_cands
        )
        offsets = torch.zeros(n_chunk, dtype=torch.int64, device=dev)
        offsets[1:] = torch.cumsum(n_cands[:-1], dim=0)
        local_idx = torch.arange(total_cands, device=dev) - offsets[tree_idx]

        sxy = span_xy[tree_idx]
        sz = span_z[tree_idx]
        area = sxy * sxy

        local_z = local_idx // area
        remainder = local_idx % area
        local_y = remainder // sxy
        local_x = remainder % sxy

        # Convert to voxel coordinates
        rri = c_root_r_int[tree_idx]
        x_vox = local_x - rri + c_xres[tree_idx]
        y_vox = local_y - rri + c_yres[tree_idx]
        z_vox = local_z

        # Validity: z < root_r_int for this tree
        valid = local_z < sz

        # Trumpet shape test
        in_root = valid & _in_trumpet_t(
            x_vox.to(torch.float64),
            y_vox.to(torch.float64),
            z_vox.to(torch.float64),
            c_xres[tree_idx].to(torch.float64),
            c_yres[tree_idx].to(torch.float64),
            torch.zeros_like(z_vox, dtype=torch.float64),        # z0 = 0
            c_above_root_z[tree_idx].to(torch.float64),          # max_height
            rri.to(torch.float64),                                # max_radius
            decay_rate,
        )

        result_x.append(x_vox[in_root].to(torch.int64))
        result_y.append(y_vox[in_root].to(torch.int64))
        result_z.append(z_vox[in_root].to(torch.int64))

    return result_x, result_y, result_z


# Segment operations

def _segment_cumsum(values, segment_ids):
    """Cumulative sum within segments — fully vectorised, no Python loop."""
    cumsum = torch.cumsum(values, dim=0)

    # Detect segment boundaries
    boundary = torch.cat([
        torch.tensor([True], device=values.device),
        segment_ids[1:] != segment_ids[:-1],
    ])
    boundary_idx = torch.nonzero(boundary, as_tuple=True)[0]

    if len(boundary_idx) <= 1:
        return cumsum

    # The offset at each boundary = cumsum at the element just before it
    # For boundary i (i>0), offset = cumsum[boundary_idx[i] - 1]
    boundary_offsets = cumsum[boundary_idx[1:] - 1]  # shape: (n_segments - 1,)

    # Map each element to its segment's offset
    # seg_labels[j] = which segment element j belongs to (0-based)
    seg_labels = torch.cumsum(boundary, dim=0) - 1  # 0, 0, ..., 1, 1, ..., 2, ...

    # Build per-element offset: segment 0 has offset 0, segment k has boundary_offsets[k-1]
    all_offsets = torch.cat([
        torch.zeros(1, dtype=cumsum.dtype, device=cumsum.device),
        boundary_offsets,
    ])  # shape: (n_segments,)

    return cumsum - all_offsets[seg_labels]


def _encode_segments(X, Y):
    x_int = (X * 1e6).to(torch.int64)
    y_int = (Y * 1e6).to(torch.int64)
    return x_int * 10_000_000_000 + y_int


# Public API

def make_voxelforest_gpu(trees_df, minx, maxx, miny, maxy,
                         vxl_per_sqm=4, stems=False, ground=True,
                         aggregation_func="max", keep=None, lad_value=1,
                         roots=False, root_ratio=0.15, decay_rate=0.4,
                         device="cuda", return_tensors=False):
    """Build a voxel forest on GPU.

    All trees of the same crown shape are processed in a single batched
    GPU kernel call — no Python per-tree loop. Supports mangrove root
    structures via the ``roots`` parameter.

    Parameters
    ----------
    trees_df : DataFrame
        Tree inventory with columns X, Y, H, CD, CL, CS, TreeID.
    minx, maxx, miny, maxy : float
        Spatial extent (metres).
    vxl_per_sqm : int
        Voxels per square metre.
    stems, ground : bool
        Include stems / ground layer.
    aggregation_func : str
        LAD aggregation for overlapping voxels.
    keep : list or None
        Extra columns to retain.
    lad_value : float
        Uniform LAD per voxel.
    roots : bool
        Include trumpet-shaped root voxels below stem (mangrove mode).
    root_ratio : float
        Root radius as fraction of tree height (root_r = H * root_ratio).
    decay_rate : float
        Exponential decay rate for trumpet root shape.
    device : str
        ``"cuda"`` or ``"cpu"``.
    return_tensors : bool
        If True, return ``dict[str, Tensor]`` instead of DataFrame.

    Returns
    -------
    DataFrame or dict[str, Tensor]
    """
    if torch is None:
        raise ImportError(
            "PyTorch is required for GPU voxel forest.  "
            "Install with: pip install torch  (or uv sync --extra gpu)"
        )

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    trees_df = pd.DataFrame(trees_df)
    vxl_sidelength = 1 / np.sqrt(vxl_per_sqm)
    vxl_per_m = np.sqrt(vxl_per_sqm)

    trees_df["Xres"] = np.round(trees_df["X"] * vxl_per_m).astype(int)
    trees_df["Yres"] = np.round(trees_df["Y"] * vxl_per_m).astype(int)
    trees_df["CDres"] = np.round(trees_df["CD"] * vxl_per_m).astype(int)
    trees_df["CRres"] = np.ceil(trees_df["CDres"] / 2).astype(int)

    all_x, all_y, all_z = [], [], []

    # Batch process all trees per crown shape (1 kernel call per CS)
    for cs in trees_df["CS"].unique():
        sub_df = trees_df[trees_df["CS"] == cs]
        if sub_df.empty:
            continue
        geom_fn, z_mode = _CS_DISPATCH[cs]

        rx, ry, rz = _batch_crown_voxels(sub_df, geom_fn, z_mode, dev)
        all_x.extend(rx)
        all_y.extend(ry)
        all_z.extend(rz)

        if stems or roots:
            sx, sy, sz = _batch_stem_voxels(sub_df, dev, roots=roots, root_ratio=root_ratio)
            all_x.extend(sx)
            all_y.extend(sy)
            all_z.extend(sz)

        if roots:
            rx, ry, rz = _batch_root_voxels(sub_df, dev, root_ratio=root_ratio, decay_rate=decay_rate)
            all_x.extend(rx)
            all_y.extend(ry)
            all_z.extend(rz)

    # Ground voxels
    if ground:
        gx_vals = torch.arange(int(minx * vxl_per_m), int(maxx * vxl_per_m) + 1, device=dev, dtype=torch.int64)
        gy_vals = torch.arange(int(miny * vxl_per_m), int(maxy * vxl_per_m) + 1, device=dev, dtype=torch.int64)
        ggx, ggy = torch.meshgrid(gx_vals, gy_vals, indexing="ij")
        all_x.append(ggx.reshape(-1))
        all_y.append(ggy.reshape(-1))
        all_z.append(torch.zeros(ggx.numel(), device=dev, dtype=torch.int64))

    if len(all_x) == 0:
        if return_tensors:
            empty = torch.zeros(0, dtype=torch.float64, device=dev)
            return {"X": empty, "Y": empty, "Z": empty, "LAD": empty, "LAI": empty}
        return pd.DataFrame(columns=["X", "Y", "Z", "LAD", "LAI"])

    cat_x = torch.cat(all_x)
    cat_y = torch.cat(all_y)
    cat_z = torch.cat(all_z)

    # --- All-GPU aggregation, sort, and LAI computation ---
    # Shift coordinates to guarantee non-negative before encoding
    x_min = cat_x.min()
    y_min = cat_y.min()
    z_min = cat_z.min()
    sx = cat_x - x_min
    sy = cat_y - y_min
    sz = cat_z - z_min

    # Determine multipliers from actual ranges to avoid collision
    y_range = (sy.max() + 1).item()
    z_range = (sz.max() + 1).item()
    voxel_key = sx * (y_range * z_range) + sy * z_range + sz

    # Deduplicate (equivalent to groupby(X,Y,Z).agg(max) with uniform LAD)
    unique_keys, inv_idx = torch.unique(voxel_key, return_inverse=True)
    n_unique = unique_keys.shape[0]

    # Decode back to shifted X, Y, Z then restore offset
    uX = unique_keys // (y_range * z_range) + x_min
    remainder = unique_keys % (y_range * z_range)
    uY = remainder // z_range + y_min
    uZ = remainder % z_range + z_min

    # For non-uniform LAD with aggregation, use scatter_reduce
    LAD_t = torch.full((n_unique,), lad_value, dtype=torch.float64, device=dev)

    # Scale back to real coordinates
    uX_real = torch.round(uX.to(torch.float64) * vxl_sidelength * 100) / 100
    uY_real = torch.round(uY.to(torch.float64) * vxl_sidelength * 100) / 100
    uZ_real = uZ.to(torch.float64)

    # Sort by X asc, Y asc, Z desc (matching CPU version)
    seg = _encode_segments(uX_real, uY_real)
    sort_key = seg.to(torch.float64) * 1e6 - uZ_real
    sort_idx = torch.argsort(sort_key)

    X_t = uX_real[sort_idx]
    Y_t = uY_real[sort_idx]
    Z_t = uZ_real[sort_idx]
    LAD_t = LAD_t[sort_idx]

    # LAI cumsum on GPU (segment-based)
    seg_sorted = _encode_segments(X_t, Y_t)
    LAI_t = _segment_cumsum(LAD_t, seg_sorted)

    if return_tensors:
        return {"X": X_t, "Y": Y_t, "Z": Z_t, "LAD": LAD_t, "LAI": LAI_t}

    vdf = pd.DataFrame({
        "X": X_t.cpu().numpy(),
        "Y": Y_t.cpu().numpy(),
        "Z": Z_t.cpu().numpy(),
        "LAD": LAD_t.cpu().numpy(),
        "LAI": LAI_t.cpu().numpy(),
    })
    return vdf
