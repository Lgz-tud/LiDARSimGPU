# LiDARSimGPU

GPU-accelerated forest voxelisation and LiDAR simulation in PyTorch.

## Introduction

LiDARSimGPU builds a 3D voxel forest from a tree inventory and simulates a
discrete LiDAR point cloud from it using a Beer-Lambert light-extinction
approach. The two core functions run fully
batched on the GPU and chain zero-copy via `return_tensors=True`. It is a GPU
reimplementation of the corresponding
routines in [slidaRtools](https://github.com/nikoknapp/slidaRtools), validated
against it.

| Function | Purpose | Docs |
|----------|---------|------|
| `make_voxelforest_gpu` | inventory -> 3D voxel forest | [voxel_forest_gpu.md](src/lidarsim_gpu/voxel_forest_gpu.md) |
| `make_lidarscan_gpu`   | voxel forest -> LiDAR point cloud | [lidar_scan_gpu.md](src/lidarsim_gpu/lidar_scan_gpu.md) |
| `display_point_cloud`  | 3D view | [display.md](src/lidarsim_gpu/display.md) |
| `point_cloud_to_chm`   | point cloud -> canopy height model | [raster.md](src/lidarsim_gpu/raster.md) |

Each function is documented in a companion `.md` next to its module, with a
structure inspired by the ODD protocol ([Grimm et al. 2006](https://doi.org/10.1016/j.ecolmodel.2006.04.023)):
overview/purpose, inputs, parameters, outputs, and validation.

## Install

Requires [uv](https://docs.astral.sh/uv/). Install uv first if needed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install the package:

```bash
uv pip install "LiDARSimGPU[viz,raster] @ git+https://github.com/Lgz-tud/LiDARSimGPU.git"
```

## Usage

```python
import pandas as pd
from lidarsim_gpu import make_voxelforest_gpu, make_lidarscan_gpu

# inventory columns: TreeID, X, Y, H, CD, CL, CS
trees = pd.read_csv("data/tree_inventory_traunstein.csv")

vxf = make_voxelforest_gpu(trees, minx=0, maxx=100, miny=0, maxy=100,
                           lad_value=0.3, device="cuda", return_tensors=True)
scan = make_lidarscan_gpu(vxf, P0_AGR=0.2, P0_GR=0.2, k=0.2, device="cuda")
```

## Demo

```bash
uv run python examples/run_demo.py
```

Reproduce the R reference benchmark:

```bash
uv run python -m pytest benchmark/ -q
```

## References

- Knapp, N., Fischer, R., Huth, A., 2018. Linking lidar and forest modeling to
  assess biomass estimation across scales and disturbance states. Remote Sens.
  Environ. 205, 199–209. https://doi.org/10.1016/j.rse.2017.11.018
- Liu, G., et al., in preparation. <!-- TODO: title, co-authors, journal, DOI on acceptance -->
- Knapp, N., Huth, A., Kugler, F., Papathanassiou, K., Condit, R., Hubbell, S.P.,
  Fischer, R., 2018. Model-assisted estimation of tropical forest biomass change.
  Remote Sens. 10, 731. https://doi.org/10.3390/rs10050731
- Rödig, E., Knapp, N., Fischer, R., Bohn, F.J., Dubayah, R., Tang, H., Huth, A.,
  2019. From small-scale forest structure to Amazon-wide carbon estimates. Nat.
  Commun. 10. https://doi.org/10.1038/s41467-019-13063-y
- Fischer, R., Knapp, N., Bohn, F., Shugart, H.H., Huth, A., 2019. The Relevance
  of Forest Structure for Biomass and Productivity in Temperate Forests. Surv.
  Geophys. https://doi.org/10.1007/s10712-019-09519-x
- Knapp, N., Huth, A., Fischer, R., 2021. Tree Crowns Cause Border Effects in
  Area-Based Biomass Estimations from Remote Sensing. Remote Sens. 13, 1592.
  https://doi.org/10.3390/rs13081592
- Bauer, L., Knapp, N., Fischer, R., 2021. Mapping Amazon Forest Productivity by
  Fusing GEDI Lidar Waveforms with an Individual-Based Forest Model. Remote Sens.
  13. https://doi.org/10.3390/rs13224540
- Bruening, J.M., Fischer, R., Bohn, F.J., Armston, J., Armstrong, A.H., Knapp, N.,
  Tang, H., Huth, A., Dubayah, R., 2021. Challenges to aboveground biomass
  prediction from waveform lidar. Environ. Res. Lett. 16, 125013.
  https://doi.org/10.1088/1748-9326/ac3cec

## License

[GPL-3](LICENSE).
