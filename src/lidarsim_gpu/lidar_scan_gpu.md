# lidar_scan_gpu.py — voxel forest to LiDAR point cloud

## Purpose

Simulate a discrete, multi-return LiDAR point cloud from a voxel forest using a
Beer-Lambert light-extinction approach, on the GPU. GPU reimplementation of
`make_lidarscan` from slidaRtools (GPL-3).

## Function

```python
make_lidarscan_gpu(vxf, P0_AGR=0.2, P0_GR=0.2, k=0.2,
                   LAD_prefactor=False, prob_out=False,
                   device="cuda", return_tensors=False)
```

## Input

`vxf` — a voxel forest, either:
- a DataFrame with columns `X, Y, Z, LAD`, or
- a tensor dict from `make_voxelforest_gpu(..., return_tensors=True)`
  (zero-copy GPU chaining).

## Key parameters

| Parameter | Meaning |
|-----------|---------|
| `P0_AGR` | above-ground initial return probability |
| `P0_GR` | ground initial return probability |
| `k` | light extinction coefficient |
| `LAD_prefactor` | multiply return probability by LAD |
| `prob_out` | also return the full voxel forest with probabilities |
| `device` | `"cuda"` or `"cpu"` |
| `return_tensors` | return `dict[str, Tensor]` instead of a DataFrame |

## Output

DataFrame with columns `X, Y, Z, LAD, LAI, ReturnProbability, ReturnNumber`
(or a tensor dict when `return_tensors=True`).

## Method

The scan is stochastic: each voxel returns with a Beer-Lambert probability that
decays with the cumulative leaf area above it. Sorting is `X` asc, `Y` asc,
`Z` desc — matching the R reference exactly.

## Validation

Because the scan is stochastic, it is validated statistically against the R
slidaRtools reference point cloud (counts, height moments, vertical profile
within tolerance); see `benchmark/test_lidarscan_vs_r.py`.
