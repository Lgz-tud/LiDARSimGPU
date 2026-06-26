# display.py — 3D view of a voxel forest or point cloud

## Purpose

Render a voxel forest or LiDAR point cloud as a coloured 3D scatter plot. Works
on the output of both `make_voxelforest_gpu` and `make_lidarscan_gpu`. Optional —
requires matplotlib (`pip install LiDARSimGPU[viz]`).

## Function

```python
display_point_cloud(xyz, col_palette=None, col_var="Z", col_lim=None,
                    size=1, filename=None)
```

## Input

`xyz` — a DataFrame with columns `X, Y, Z`, or the tensor dict returned with
`return_tensors=True` (converted to CPU automatically).

## Key parameters

| Parameter | Meaning |
|-----------|---------|
| `col_palette` | matplotlib colormap (default `jet`) |
| `col_var` | column used for colouring (default `"Z"` = height) |
| `col_lim` | colour limits `[min, max]` (default data range) |
| `size` | marker size |
| `filename` | save to this path instead of showing interactively (use on headless machines / in CI) |

## Output

Shows a window, or writes a PNG when `filename` is given.
