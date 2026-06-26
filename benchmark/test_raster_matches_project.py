"""Lock package CHM output to the research pipeline's output.

The package's rasterisation (lidarsim_gpu.raster) was copied verbatim from
research/paper_a/analysis/01_process_pointcloud.py. This test embeds a FROZEN
snapshot of that pipeline's array-producing functions and asserts that the
package reproduces them value-for-value on the real Traunstein LiDAR point
cloud, across all CHM resolutions, plus the empty-cloud edge case.

If anyone edits lidarsim_gpu.raster and changes the numbers, this test fails.

DO NOT EDIT the _ref_* functions below: they are a verbatim regression anchor
to the published research pipeline, not a place for improvements.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from rasterio.transform import from_origin

from lidarsim_gpu.raster import point_cloud_to_chm


# FROZEN reference — verbatim from 01_process_pointcloud.py (research pipeline)

def _ref_snap_extent_to_grid(xmin, xmax, ymin, ymax, res=1.0, origin=(0.0, 0.0)):
    x0, y0 = origin
    gxmin = x0 + np.floor((xmin - x0) / res) * res
    gymax = y0 + np.ceil((ymax - y0) / res) * res
    gxmax = x0 + np.ceil((xmax - x0) / res) * res
    gymin = y0 + np.floor((ymin - y0) / res) * res
    return gxmin, gxmax, gymin, gymax


def _ref_raster_max_aggregate(df, transform, width, height, z_filter_min=0.0):
    grid = np.full((height, width), -np.inf, dtype=np.float32)
    west, north = transform.c, transform.f
    res_x, res_y = transform.a, -transform.e
    X = df["X"].to_numpy(np.float64)
    Y = df["Y"].to_numpy(np.float64)
    Z = df["Z"].to_numpy(np.float32)
    m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    if z_filter_min is not None:
        m &= Z >= float(z_filter_min)
    if not np.any(m):
        grid[~np.isfinite(grid)] = np.float32(0.0)
        return grid
    cols = np.floor((X[m] - west) / res_x).astype(np.int64)
    rows = np.floor((north - Y[m]) / res_y).astype(np.int64)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if np.any(inside):
        idx = (rows[inside] * width + cols[inside]).astype(np.int64)
        np.maximum.at(grid.ravel(), idx, Z[m][inside])
    grid[~np.isfinite(grid)] = np.float32(0.0)
    return grid


def _ref_block_reduce_stat(a, factor, *, nodata=-9999.0, stat="p95", tile=2048):
    import math
    a = a.astype(np.float32, copy=False)
    a = np.where(a == nodata, np.nan, a)
    H, W = a.shape
    H2 = math.ceil(H / factor) * factor
    W2 = math.ceil(W / factor) * factor
    if H2 != H or W2 != W:
        padded = np.full((H2, W2), np.nan, dtype=np.float32)
        padded[:H, :W] = a
        a = padded
    nh = H2 // factor
    nw = W2 // factor
    out = np.empty((nh, nw), dtype=np.float32)
    step = max(1, tile // factor)
    for r0 in range(0, nh, step):
        r1 = min(nh, r0 + step)
        sub = a[r0 * factor: r1 * factor, :W2].reshape(r1 - r0, factor, nw, factor)
        if stat == "max":
            out[r0:r1, :] = np.nanmax(sub, axis=(1, 3))
        elif stat == "mean":
            out[r0:r1, :] = np.nanmean(sub, axis=(1, 3))
        elif stat.startswith("p"):
            q = float(stat[1:])
            flat = sub.reshape((r1 - r0) * nw, factor * factor)
            out[r0:r1, :] = np.nanpercentile(flat, q, axis=1).reshape(r1 - r0, nw)
        else:
            out[r0:r1, :] = np.nanmax(sub, axis=(1, 3))
    return np.where(np.isnan(out), nodata, out).astype(np.float32)


def _ref_build_chm(df, domain_xmax, domain_ymax, resolutions, origin=(0.0, 0.0),
                   z_min=0.0):
    """Verbatim build logic from build_chm_for_file (research pipeline)."""
    x0, y0 = origin
    gxmin, gxmax, gymin, gymax = _ref_snap_extent_to_grid(
        0, domain_xmax, 0, domain_ymax, 1.0, (x0, y0))
    width = int(round((gxmax - gxmin) / 1.0))
    height = int(round((gymax - gymin) / 1.0))
    transform_1m = from_origin(gxmin, gymax, 1.0, 1.0)
    base = _ref_raster_max_aggregate(df, transform_1m, width, height, z_filter_min=z_min)
    stat_map = {int(r): "max" for r in resolutions}
    out = {}
    for res in map(float, resolutions):
        if abs(res - 1.0) < 1e-9:
            grid, tfm = base, transform_1m
        else:
            factor = int(round(res / 1.0))
            stat = stat_map.get(int(res), "p95")
            grid = _ref_block_reduce_stat(base, factor, stat=stat)
            tfm = transform_1m * Affine.scale(factor)
        out[res] = (grid, tfm)
    return out


RESOLUTIONS = (1, 2, 5, 10, 20, 50, 100)

DATA = Path(__file__).resolve().parent.parent / "data" / "lidar_pointcloud_traunstein.csv"


@pytest.fixture(scope="module")
def cloud():
    """Real Traunstein LiDAR point cloud (X, Y, Z)."""
    return pd.read_csv(DATA)


class TestRasterMatchesProject:
    """Package CHM must equal the frozen research-pipeline CHM, value for value."""

    def test_all_resolutions_identical(self, cloud):
        pkg = point_cloud_to_chm(cloud, 100, 100, resolutions=RESOLUTIONS)
        ref = _ref_build_chm(cloud, 100, 100, RESOLUTIONS)

        assert set(pkg.keys()) == set(ref.keys())
        for res in RESOLUTIONS:
            g_pkg, t_pkg = pkg[float(res)]
            g_ref, t_ref = ref[float(res)]
            assert g_pkg.shape == g_ref.shape, f"{res} m shape differs"
            assert np.array_equal(g_pkg, g_ref), f"{res} m grid values differ"
            assert tuple(t_pkg)[:6] == tuple(t_ref)[:6], f"{res} m transform differs"

    def test_empty_cloud_all_zero(self):
        empty = pd.DataFrame({"X": [], "Y": [], "Z": []})
        pkg = point_cloud_to_chm(empty, 100, 100, resolutions=(1,))
        grid, _ = pkg[1.0]
        assert np.all(grid == 0.0), "Empty cloud must give an all-zero CHM"
