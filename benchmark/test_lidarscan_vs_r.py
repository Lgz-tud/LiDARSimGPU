"""make_lidarscan_gpu must statistically reproduce the R slidaRtools point cloud.

The LiDAR scan is stochastic (Beer-Lambert return probabilities), so the match
is statistical, not exact. To isolate the scan step, the GPU scan is run on the
*R reference voxel forest* and compared against the R reference point cloud, both
generated with P0.AGR=0.2, P0.GR=0.2, k=0.2.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

# Statistical tolerances — stochastic process, so allow small deviations.
TOL_COUNT_PCT = 3.0     # total return count within 3 %
TOL_Z_MEAN = 0.5        # mean canopy height within 0.5 m
TOL_Z_STD = 0.5         # height spread within 0.5 m
TOL_BIN_PROP = 0.02     # per-1m-bin proportion within 2 percentage points


@pytest.fixture(scope="session")
def gpu_pointcloud(r_voxelforest):
    """GPU scan of the R reference voxel forest (same parameters as R)."""
    from lidarsim_gpu import make_lidarscan_gpu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return make_lidarscan_gpu(
        r_voxelforest, P0_AGR=0.2, P0_GR=0.2, k=0.2, device=device,
    )


class TestLidarScanVsR:
    """GPU scan distribution must match the R reference point cloud."""

    def test_total_count(self, r_pointcloud, gpu_pointcloud):
        n_r, n_gpu = len(r_pointcloud), len(gpu_pointcloud)
        pct = abs(n_gpu - n_r) / n_r * 100
        assert pct < TOL_COUNT_PCT, f"Count diff {pct:.2f}% (R={n_r}, GPU={n_gpu})"

    def test_z_mean(self, r_pointcloud, gpu_pointcloud):
        diff = abs(gpu_pointcloud["Z"].mean() - r_pointcloud["Z"].mean())
        assert diff < TOL_Z_MEAN, f"Z mean diff {diff:.4f} m"

    def test_z_std(self, r_pointcloud, gpu_pointcloud):
        diff = abs(gpu_pointcloud["Z"].std() - r_pointcloud["Z"].std())
        assert diff < TOL_Z_STD, f"Z std diff {diff:.4f} m"

    def test_z_max(self, r_pointcloud, gpu_pointcloud):
        diff = abs(gpu_pointcloud["Z"].max() - r_pointcloud["Z"].max())
        assert diff <= 1.0, f"Z max diff {diff:.4f} m"

    def test_ground_fraction(self, r_pointcloud, gpu_pointcloud):
        f_r = (r_pointcloud["Z"] == 0).mean()
        f_gpu = (gpu_pointcloud["Z"] == 0).mean()
        assert abs(f_gpu - f_r) < 0.03, f"Ground fraction R={f_r:.3f}, GPU={f_gpu:.3f}"

    def test_z_profile_shape(self, r_pointcloud, gpu_pointcloud):
        """Vertical return profile (1 m bins, normalised) must match in shape."""
        z_max = max(r_pointcloud["Z"].max(), gpu_pointcloud["Z"].max())
        bins = np.arange(0, z_max + 2)
        p_r, _ = np.histogram(r_pointcloud["Z"], bins=bins, density=False)
        p_gpu, _ = np.histogram(gpu_pointcloud["Z"], bins=bins, density=False)
        p_r = p_r / p_r.sum()
        p_gpu = p_gpu / p_gpu.sum()
        max_dev = np.abs(p_r - p_gpu).max()
        assert max_dev < TOL_BIN_PROP, f"Max per-bin proportion deviation {max_dev:.4f}"
