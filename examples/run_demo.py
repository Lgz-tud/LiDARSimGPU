"""End-to-end demo for LiDARSimGPU.

Runs the two GPU functions on the Traunstein example inventory in a zero-copy
chain and saves 3D views of the voxel forest and the simulated point cloud.
Falls back to CPU when no CUDA device is available.

Run:
    uv run python examples/run_demo.py
"""

from pathlib import Path

import pandas as pd

from lidarsim_gpu import (
    make_voxelforest_gpu,
    make_lidarscan_gpu,
    display_point_cloud,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "tree_inventory_traunstein.csv"


def main() -> None:
    trees = pd.read_csv(DATA)
    print(f"Traunstein inventory: {len(trees)} trees")

    out_dir = Path(__file__).resolve().parent / "simulation_output"
    out_dir.mkdir(exist_ok=True)

    # 1. Inventory -> 3D voxel forest. return_tensors=True keeps the output on
    #    the GPU as a tensor dict for zero-copy chaining into the scan.
    vxf = make_voxelforest_gpu(
        trees, minx=0, maxx=100, miny=0, maxy=100,
        vxl_per_sqm=4, stems=True, ground=True, lad_value=0.3,
        device="cuda", return_tensors=True,
    )
    print(f"Voxel forest: {len(vxf['X'])} voxels")

    # 2. Voxel forest -> simulated LiDAR point cloud (multi-return Beer-Lambert).
    scan = make_lidarscan_gpu(vxf, P0_AGR=0.2, P0_GR=0.2, k=0.2, device="cuda")
    print(f"Simulated LiDAR returns: {len(scan)} points")

    # 3. Save 3D views of both the voxel forest and the simulated point cloud.
    display_point_cloud(vxf, size=1, filename=out_dir / "voxel_forest.png")
    display_point_cloud(scan, size=2, filename=out_dir / "lidar_pointcloud.png")
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
