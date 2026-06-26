# voxel_forest_gpu.py — tree inventory to 3D voxel forest

## Purpose

Build a 3D voxel forest from a tree inventory on the GPU. All trees of the same
crown shape are processed in a single batched PyTorch kernel call (no per-tree
Python loop). GPU reimplementation of `make_voxelforest` from slidaRtools (GPL-3).

## Function

```python
make_voxelforest_gpu(trees_df, minx, maxx, miny, maxy,
                     vxl_per_sqm=4, stems=False, ground=True,
                     aggregation_func="max", keep=None, lad_value=1,
                     roots=False, root_ratio=0.15, decay_rate=0.4,
                     device="cuda", return_tensors=False)
```

## Input

Inventory DataFrame with columns:

| Column | Meaning |
|--------|---------|
| `TreeID` | tree id |
| `X`, `Y` | stem position (m) |
| `H` | tree height (m) |
| `CD` | crown diameter (m) |
| `CL` | crown length (m) |
| `CS` | crown shape code |

Crown shape codes (`CS`): 1 cylinder, 2 spheroid, 3 cone, 4 ice-cone,
5 inverted cone.

## Key parameters

| Parameter | Meaning |
|-----------|---------|
| `vxl_per_sqm` | voxels per m² (horizontal resolution) |
| `stems`, `ground` | include stem column / ground layer |
| `lad_value` | uniform leaf area density per voxel |
| `roots` | add trumpet-shaped root voxels below the stem (mangrove mode) |
| `device` | `"cuda"` or `"cpu"` (falls back to CPU if no GPU) |
| `return_tensors` | return `dict[str, Tensor]` for zero-copy chaining into the scan |

## Output

DataFrame `(X, Y, Z, LAD, LAI)` — or a tensor dict when `return_tensors=True`.

## Validation

Voxel forest is deterministic and reproduces the R slidaRtools reference
exactly (spatial Jaccard = 1.0, identical Z histogram and LAD); see
`benchmark/test_voxelforest_vs_r.py`.
