# Rasterisation — point cloud to Canopy Height Model (CHM)

Documentation for `lidarsim_gpu.raster`. The functions here are copied verbatim
from the research pipeline (`01_process_pointcloud.py`) so that CHMs produced by
this package are bit-for-bit identical to the published results. The regression
test `benchmark/test_raster_matches_project.py` locks this identity.

This page documents only the CHM-generation core. The research pipeline also
computes plot-level remote-sensing metrics and runs a parallel multi-scenario
workflow; those parts are project-specific and are **not** included here.

## Functions

| Function | Purpose |
|----------|---------|
| `snap_extent_to_grid` | Snap an extent to a fixed grid (origin-aligned) |
| `raster_max_aggregate` | Rasterise a point cloud to a max-height grid |
| `block_reduce_stat` | Aggregate a fine grid to a coarser one (max / mean / percentile) |
| `write_geotiff` | Write a grid to a (compressed, tiled) GeoTIFF |
| `point_cloud_to_chm` | High-level: point cloud -> multi-resolution CHM grids |

## CHM methodology (Knapp et al. 2018)

Following Knapp et al. (2018, *Remote Sensing of Environment* 205):

- **1 m CHM**: `max(point cloud Z)` per 1 m pixel.
- **Coarse CHM** (2 m / 5 m / ...): `max(1 m pixels in block)` — equivalent to
  direct rasterisation at the coarser resolution.
- **Empty pixels**: filled with 0 m (a gap), not nodata, following Asner &
  Mascaro (2014) and the slidaRtools `gapfill(fill.value=0)` behaviour.

### Aggregation settings

- `STAT_MAP`: **max** at all resolutions (CHM = top-of-canopy height).
- `block_reduce_stat`: edges are **NaN-padded** (not truncated) so no plot area
  is lost when the grid size is not a multiple of the aggregation factor.
- Fixed domain extent: the grid is snapped to a supplied domain extent, not to
  the data bounds, so different point clouds produce same-size, aligned rasters.

## Validation against R / lidR

The `np.maximum.at` rasterisation was independently validated against
`lidR::rasterize_canopy(p2r())` in the research project's `r_validation`:

- CHM (1 m, max per pixel): **100 % exact match** with lidR `p2r()`.

This is the basis for treating the package CHM as a faithful, reproducible
implementation of the standard max-height CHM.

## Usage

```python
from lidarsim_gpu import make_lidarscan_gpu, point_cloud_to_chm

scan = make_lidarscan_gpu(vxf, P0_AGR=0.2, P0_GR=0.2, k=0.2)

# multi-resolution CHM over a fixed 100 x 100 m domain
chm = point_cloud_to_chm(scan, domain_xmax=100, domain_ymax=100,
                         resolutions=(1, 2, 5, 10), out_dir="chm_out")
grid_1m, transform_1m = chm[1.0]
```

`out_dir` (optional) writes `{stem}_chm_{res}m.tif` GeoTIFFs.

## References

- Knapp, N., Fischer, R., Huth, A., 2018. Linking lidar and forest modeling to
  assess biomass estimation across scales and disturbance states. Remote Sens.
  Environ. 205, 199–209. https://doi.org/10.1016/j.rse.2017.11.018
- Asner, G.P., Mascaro, J., 2014. Mapping tropical forest carbon: Calibrating
  plot estimates to a simple LiDAR metric. Remote Sens. Environ. 140, 614–624.
  https://doi.org/10.1016/j.rse.2013.09.023
