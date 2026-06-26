# Point cloud -> Canopy Height Model (CHM) rasterisation.
#
# The four core functions below (snap_extent_to_grid, raster_max_aggregate,
# block_reduce_stat, write_geotiff) are copied VERBATIM from the research
# pipeline research/paper_a/analysis/01_process_pointcloud.py so that the CHM
# produced by this package is bit-for-bit identical to the one used in the
# paper. Do not "improve" them: any change risks diverging from the published
# results. The regression test benchmark/test_raster_matches_project.py locks
# this identity.
#
# CHM convention: pixel value = max return height; empty cells = 0 m (a gap),
# following Asner & Mascaro (2014, RSE) and the slidaRtools gapfill(fill.value=0)
# behaviour. CHM = max per pixel at all resolutions (Knapp et al. 2018, RSE).

import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.transform import from_origin


def snap_extent_to_grid(xmin, xmax, ymin, ymax, res=1.0, origin=(0.0, 0.0)):
    x0, y0 = origin
    gxmin = x0 + np.floor((xmin - x0) / res) * res
    gymax = y0 + np.ceil((ymax - y0) / res) * res
    gxmax = x0 + np.ceil((xmax - x0) / res) * res
    gymin = y0 + np.floor((ymin - y0) / res) * res
    return gxmin, gxmax, gymin, gymax


def raster_max_aggregate(df: pd.DataFrame, transform, width: int, height: int,
                         z_filter_min: Optional[float] = 0.0) -> np.ndarray:
    """Rasterize point cloud to max-height grid."""
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
        # Asner & Mascaro 2014 RSE / Knapp slidaRtools gapfill(fill.value=0):
        # empty cells = 0 m (gap), not nodata. Lets TCH = mean over total plot
        # area instead of canopy-only mean.
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


def block_reduce_stat(a, factor, *, nodata=-9999.0, stat="p95", tile=2048):
    """Reduce CHM grid by factor using stat aggregation."""
    a = a.astype(np.float32, copy=False)
    a = np.where(a == nodata, np.nan, a)
    H, W = a.shape

    # Pad to nearest multiple of factor (NaN fill, ignored by nan-aware funcs)
    import math
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


def write_geotiff(out_path: Path, array: np.ndarray, transform,
                  crs=None, compress="lzw", nodata=-9999.0):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype="float32", transform=transform,
        tiled=True, compress=compress, nodata=nodata,
        BIGTIFF="IF_NEEDED", predictor=2,
    )
    if crs:
        profile["crs"] = crs
    with rasterio.open(out_path.as_posix(), "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)


# DataFrame -> multi-resolution CHM (replicates build_chm_for_file from the
# research pipeline, but takes an in-memory point cloud + explicit domain
# instead of reading an h5 file and module-level config globals).

def point_cloud_to_chm(
    df: pd.DataFrame,
    domain_xmax: float,
    domain_ymax: float,
    resolutions: Tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100),
    origin: Tuple[float, float] = (0.0, 0.0),
    z_min: float = 0.0,
    stat_map: Optional[Dict[int, str]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    stem: str = "chm",
    crs=None,
    compress: str = "lzw",
) -> Dict[float, Tuple[np.ndarray, Affine]]:
    """Build multi-resolution CHM grids from a point cloud DataFrame.

    This reproduces ``build_chm_for_file`` from the research pipeline exactly:
    a 1 m base grid is built with ``raster_max_aggregate`` over the fixed domain
    extent, then coarser resolutions are produced by ``block_reduce_stat`` (max
    aggregation by default). Output therefore matches the paper's CHM.

    Parameters
    ----------
    df : DataFrame
        Point cloud with columns X, Y, Z (e.g. ``make_lidarscan_gpu`` output).
    domain_xmax, domain_ymax : float
        Fixed domain extent (metres). The grid is snapped to this extent, NOT
        to the data bounds, so CHMs of different point clouds are comparable.
    resolutions : tuple of float
        CHM pixel sizes (metres). 1 m is the base; coarser sizes are aggregated.
    origin : (float, float)
        Grid origin (default (0, 0)).
    z_min : float
        Minimum return height kept when rasterising (default 0.0).
    stat_map : dict or None
        Per-resolution aggregation statistic. Defaults to "max" for every
        resolution (Knapp et al. 2018); resolutions absent from the map fall
        back to "p95", matching the research pipeline's ``STAT_MAP.get``.
    out_dir : path or None
        If given, also write ``{stem}_chm_{res}m.tif`` GeoTIFFs there.
    stem : str
        Filename stem for written GeoTIFFs.
    crs, compress :
        Passed through to ``write_geotiff``.

    Returns
    -------
    dict[float, (ndarray, Affine)]
        Maps each resolution to its CHM grid and affine transform.
    """
    if stat_map is None:
        stat_map = {int(r): "max" for r in resolutions}

    x0, y0 = origin
    gxmin, gxmax, gymin, gymax = snap_extent_to_grid(
        0, domain_xmax, 0, domain_ymax, 1.0, (x0, y0))
    width = int(round((gxmax - gxmin) / 1.0))
    height = int(round((gymax - gymin) / 1.0))
    transform_1m = from_origin(gxmin, gymax, 1.0, 1.0)

    base = raster_max_aggregate(df, transform_1m, width, height, z_filter_min=z_min)

    out: Dict[float, Tuple[np.ndarray, Affine]] = {}
    for res in map(float, resolutions):
        if abs(res - 1.0) < 1e-9:
            grid, tfm = base, transform_1m
        else:
            factor = int(round(res / 1.0))
            stat = stat_map.get(int(res), "p95")
            grid = block_reduce_stat(base, factor, stat=stat)
            tfm = transform_1m * Affine.scale(factor)
        out[res] = (grid, tfm)

        if out_dir is not None:
            tag = f"{int(res)}" if float(res).is_integer() else f"{res}"
            write_geotiff(Path(out_dir) / f"{stem}_chm_{tag}m.tif",
                          grid, tfm, crs=crs, compress=compress)

    return out
