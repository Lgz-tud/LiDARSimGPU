# Data provenance and attribution

The two CSV files in this directory are derived from the demonstration dataset
shipped with the R package **slidaRtools**:

- File: `DemoWorkflow/Demo_Data_1ha_Traunstein.rda`
- Author: **Nikolai Knapp**
- Copyright: **© 2017 Helmholtz-Zentrum für Umweltforschung GmbH – UFZ**
- Repository: https://github.com/niknap/slidaRtools
- License: **GPL-3** (same as this package)

## Files

| File | Source object | Description |
|------|---------------|-------------|
| `tree_inventory_traunstein.csv`   | `inv.dt` | 1 ha forest inventory, Traunstein (Bavaria, Germany). Crown attributes (H, CL, CD, CS, LAD) derived from DBH with the slidaRtools demo-workflow equations. |
| `lidar_pointcloud_traunstein.csv` | `lid.dt` | Real airborne LiDAR point cloud over the same 1 ha plot. |

The derived crown attributes follow the slidaRtools demo workflow
(`Demo_Workflow_LidarSimulation.R`):

```
H  = (60 * DBH) / (0.5 + DBH)   # Michaelis-Menten tree height
CL = 0.4 * H                    # crown length
CD = 15 * DBH^0.8               # crown diameter (power law)
CS = 4                          # crown shape: ice-cone
LAD = 0.3                       # leaf area density [m2 m-3]
```

## Citation

If you use this dataset, please cite the original sources:

- Knapp, N., Fischer, R., & Huth, A. (2018). Linking lidar and forest modeling
  to assess biomass estimation across scales and disturbance states.
  *Remote Sensing of Environment*, 205, 199-209.
- Knapp, N., Huth, A., & Fischer, R. (2021). Tree crowns cause border effects
  in area-based biomass estimations from remote sensing.
  *Remote Sensing*, 13(8), 1592.
