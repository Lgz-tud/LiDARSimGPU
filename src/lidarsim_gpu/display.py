# Plot a voxel forest or LiDAR point cloud in 3D space.

# Works on the output of both make_voxelforest_gpu and make_lidarscan_gpu,
# which are XYZ tables. Accepts a pandas DataFrame or the tensor dict
# returned with return_tensors=True.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def _to_dataframe(xyz):
    # Accept a tensor dict (from return_tensors=True) or a DataFrame.
    if isinstance(xyz, dict):
        # Detach tensors to CPU numpy if torch tensors are present
        data = {}
        for key, val in xyz.items():
            data[key] = val.cpu().numpy() if hasattr(val, "cpu") else np.asarray(val)
        return pd.DataFrame(data)
    return pd.DataFrame(xyz)


def display_point_cloud(xyz, col_palette=None, col_var="Z", col_lim=None,
                        size=1, filename=None):
    """Render a voxel forest or point cloud as a coloured 3D scatter.

    Parameters
    ----------
    xyz : DataFrame or dict[str, Tensor]
        Output of make_voxelforest_gpu or make_lidarscan_gpu (columns X, Y, Z).
    col_palette : matplotlib Colormap or None
        Colour map; defaults to ``jet``.
    col_var : str
        Column used for colouring (default ``"Z"`` = height).
    col_lim : list[float] or None
        Colour limits; defaults to data min/max.
    size : float
        Marker size.
    filename : str or None
        If given, save the figure to this path instead of showing it
        interactively. Use this on headless machines / in CI.
    """
    xyz_dt = _to_dataframe(xyz)

    # Default blue-to-red heat palette
    if col_palette is None:
        col_palette = plt.get_cmap("jet")

    # Stretch colour limits between min and max of the colour variable
    if col_lim is None:
        col_lim = [np.floor(xyz_dt[col_var].min()),
                   np.ceil(xyz_dt[col_var].max())]

    norm = Normalize(vmin=col_lim[0], vmax=col_lim[1])
    sm = ScalarMappable(cmap=col_palette, norm=norm)
    col_index = sm.to_rgba(xyz_dt[col_var])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    ax.scatter(xyz_dt["X"], xyz_dt["Y"], xyz_dt["Z"], c=col_index, s=size)
    ax.set_axis_off()

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig)
    else:
        plt.show()
