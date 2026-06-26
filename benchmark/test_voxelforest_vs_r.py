"""make_voxelforest_gpu must reproduce the R slidaRtools reference exactly.

Parameters match the reference generation (see r_reference/README.md):
    make_voxelforest(inv, minx=0, maxx=100, miny=0, maxy=100,
                     vxl_per_sqm=4, stems=FALSE, ground=TRUE, lad_value=0.3)
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


@pytest.fixture(scope="session")
def gpu_voxelforest(tree_inventory):
    """GPU voxel forest with the same parameters as the R reference."""
    from lidarsim_gpu import make_voxelforest_gpu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return make_voxelforest_gpu(
        tree_inventory,
        minx=0, maxx=100, miny=0, maxy=100,
        vxl_per_sqm=4, stems=False, ground=True, lad_value=0.3,
        device=device,
    )


class TestVoxelForestVsR:
    """GPU voxel forest must exactly match the R reference."""

    def test_row_count(self, r_voxelforest, gpu_voxelforest):
        assert len(gpu_voxelforest) == len(r_voxelforest), (
            f"Row count: GPU={len(gpu_voxelforest)}, R={len(r_voxelforest)}"
        )

    def test_ground_voxel_count(self, r_voxelforest, gpu_voxelforest):
        n_r = (r_voxelforest["Z"] == 0).sum()
        n_gpu = (gpu_voxelforest["Z"] == 0).sum()
        assert n_gpu == n_r, f"Ground voxels: GPU={n_gpu}, R={n_r}"

    def test_canopy_voxel_count(self, r_voxelforest, gpu_voxelforest):
        n_r = (r_voxelforest["Z"] > 0).sum()
        n_gpu = (gpu_voxelforest["Z"] > 0).sum()
        assert n_gpu == n_r, f"Canopy voxels: GPU={n_gpu}, R={n_r}"

    def test_z_max(self, r_voxelforest, gpu_voxelforest):
        assert gpu_voxelforest["Z"].max() == r_voxelforest["Z"].max()

    def test_z_mean(self, r_voxelforest, gpu_voxelforest):
        np.testing.assert_allclose(
            gpu_voxelforest["Z"].mean(), r_voxelforest["Z"].mean(), atol=1e-4,
        )

    def test_spatial_jaccard_perfect(self, r_voxelforest, gpu_voxelforest):
        """Every (X, Y, Z) in R must exist in GPU and vice versa."""
        key = ["X", "Y", "Z"]
        merged = pd.merge(
            r_voxelforest[key].drop_duplicates(),
            gpu_voxelforest[key].drop_duplicates(),
            on=key, how="outer", indicator=True,
        )
        jaccard = (merged["_merge"] == "both").sum() / len(merged)
        assert jaccard == 1.0, (
            f"Jaccard={jaccard:.6f}, "
            f"R_only={(merged['_merge'] == 'left_only').sum()}, "
            f"GPU_only={(merged['_merge'] == 'right_only').sum()}"
        )

    def test_lad_values_match(self, r_voxelforest, gpu_voxelforest):
        key = ["X", "Y", "Z"]
        merged = pd.merge(
            r_voxelforest[key + ["LAD"]],
            gpu_voxelforest[key + ["LAD"]],
            on=key, suffixes=("_r", "_gpu"),
        )
        canopy = merged[merged["Z"] > 0]
        np.testing.assert_allclose(
            canopy["LAD_gpu"].values, canopy["LAD_r"].values, atol=1e-6,
            err_msg="Canopy LAD differs between R and GPU",
        )

    def test_z_histogram_exact(self, r_voxelforest, gpu_voxelforest):
        z_max = max(r_voxelforest["Z"].max(), gpu_voxelforest["Z"].max())
        bins = np.arange(0, z_max + 2)
        hist_r, _ = np.histogram(r_voxelforest["Z"], bins=bins)
        hist_gpu, _ = np.histogram(gpu_voxelforest["Z"], bins=bins)
        np.testing.assert_array_equal(hist_gpu, hist_r, err_msg="Z histogram mismatch")
