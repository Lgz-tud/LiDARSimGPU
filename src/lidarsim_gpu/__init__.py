"""LiDARSimGPU — GPU-accelerated forest voxelisation and LiDAR simulation.

A minimal, citable distribution of the two GPU functions used in the
accompanying paper:

- :func:`make_voxelforest_gpu` — tree inventory -> 3D voxel forest (PyTorch)
- :func:`make_lidarscan_gpu`   — voxel forest -> Beer-Lambert LiDAR point cloud
- :func:`display_point_cloud`  — 3D scatter view of either output (matplotlib)

The two core functions are fully self-contained (only numpy / pandas / torch)
and can be chained zero-copy on the GPU via ``return_tensors=True``. The
visualisation helper additionally requires matplotlib (``pip install
LiDARSimGPU[viz]``).

This package is a GPU reimplementation derived from the R package
``slidaRtools`` (author Nikolai Knapp; (C) 2017 Helmholtz-Zentrum fuer
Umweltforschung GmbH - UFZ), GPL-3. See LICENSE.
"""

from lidarsim_gpu.voxel_forest_gpu import make_voxelforest_gpu
from lidarsim_gpu.lidar_scan_gpu import make_lidarscan_gpu

__version__ = "0.1.0"

__all__ = [
    "make_voxelforest_gpu",
    "make_lidarscan_gpu",
]

# Visualisation is optional — only exposed if matplotlib is installed, so the
# core simulation functions stay importable on a minimal numpy/pandas/torch env.
try:
    from lidarsim_gpu.display import display_point_cloud
    __all__.append("display_point_cloud")
except ImportError:
    pass

# Rasterisation (CHM) is optional — needs rasterio. Output is bit-identical to
# the research pipeline (see lidarsim_gpu/raster.py).
try:
    from lidarsim_gpu.raster import point_cloud_to_chm
    __all__.append("point_cloud_to_chm")
except ImportError:
    pass
