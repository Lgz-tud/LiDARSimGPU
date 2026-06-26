# LiDARSimGPU — derived from slidaRtools (author Nikolai Knapp;
# (C) 2017 Helmholtz-Zentrum fuer Umweltforschung GmbH - UFZ), GPL-3.
# This is a modified, GPU-accelerated reimplementation. Per GPL-3 (5),
# this file is a changed version of the original algorithm. See LICENSE.
"""GPU-accelerated LiDAR scan simulation using PyTorch.

Produces output identical to :func:`make_lidarscan` but replaces pandas
groupby operations with vectorised tensor operations on GPU (or CPU).

Supports tensor dict input (from :func:`make_voxelforest_gpu` with
``return_tensors=True``) for zero-copy GPU chaining.
"""

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None


# Segment operations (self-contained — duplicated from voxel_forest_gpu so
# this module has no internal package dependencies)

def _encode_segments(X, Y):
    x_int = (X * 1e6).to(torch.int64)
    y_int = (Y * 1e6).to(torch.int64)
    return x_int * 10_000_000_000 + y_int


def _segment_cumsum(values, segment_ids):
    """Cumulative sum within segments — fully vectorised, no Python loop."""
    cumsum = torch.cumsum(values, dim=0)
    boundary = torch.cat([
        torch.tensor([True], device=values.device),
        segment_ids[1:] != segment_ids[:-1],
    ])
    boundary_idx = torch.nonzero(boundary, as_tuple=True)[0]
    if len(boundary_idx) <= 1:
        return cumsum
    boundary_offsets = cumsum[boundary_idx[1:] - 1]
    seg_labels = torch.cumsum(boundary, dim=0) - 1
    all_offsets = torch.cat([
        torch.zeros(1, dtype=cumsum.dtype, device=cumsum.device),
        boundary_offsets,
    ])
    return cumsum - all_offsets[seg_labels]


def _segment_cumcount(segment_ids):
    """Cumulative count (1-based) within segments — fully vectorised."""
    ones = torch.ones(len(segment_ids), dtype=torch.int64, device=segment_ids.device)
    cumcount = torch.cumsum(ones, dim=0)
    boundary = torch.cat([
        torch.tensor([True], device=segment_ids.device),
        segment_ids[1:] != segment_ids[:-1],
    ])
    boundary_idx = torch.nonzero(boundary, as_tuple=True)[0]
    if len(boundary_idx) <= 1:
        return cumcount
    # Offset at each boundary = cumcount[boundary_start] - 1
    boundary_offsets = cumcount[boundary_idx[1:]] - 1
    seg_labels = torch.cumsum(boundary, dim=0) - 1
    all_offsets = torch.cat([
        torch.zeros(1, dtype=cumcount.dtype, device=cumcount.device),
        boundary_offsets,
    ])
    return cumcount - all_offsets[seg_labels]


def _parse_input(vxf, device):
    """Accept DataFrame or tensor dict, return X, Y, Z, LAD tensors on device."""
    dev = torch.device(device)

    if isinstance(vxf, dict) and "X" in vxf and isinstance(vxf["X"], torch.Tensor):
        # Already tensor dict — zero-copy if same device
        X = vxf["X"].to(dev)
        Y = vxf["Y"].to(dev)
        Z = vxf["Z"].to(dev)
        LAD = vxf["LAD"].to(dev)
        return X, Y, Z, LAD, True  # already_sorted flag

    # DataFrame path
    df = pd.DataFrame(vxf)
    df = df.sort_values(by=['X', 'Y', 'Z'], ascending=[True, True, False]).reset_index(drop=True)
    X = torch.tensor(df['X'].values, dtype=torch.float64, device=dev)
    Y = torch.tensor(df['Y'].values, dtype=torch.float64, device=dev)
    Z = torch.tensor(df['Z'].values, dtype=torch.float64, device=dev)
    LAD = torch.tensor(df['LAD'].values, dtype=torch.float64, device=dev)
    return X, Y, Z, LAD, True


# Public API

def make_lidarscan_gpu(vxf, P0_AGR=0.2, P0_GR=0.2, k=0.2,
                       LAD_prefactor=False, prob_out=False,
                       device="cuda", return_tensors=False):
    """GPU-accelerated LiDAR scan simulation (multi-return, Beer-Lambert).

    Parameters
    ----------
    vxf : DataFrame or dict[str, Tensor]
        Voxel forest. Accepts DataFrame (X, Y, Z, LAD) or tensor dict from
        ``make_voxelforest_gpu(..., return_tensors=True)`` for zero-copy.
    P0_AGR, P0_GR : float
        Initial return probabilities.
    k : float
        Light extinction coefficient.
    LAD_prefactor : bool
        Multiply probability by LAD.
    prob_out : bool
        Also return full voxel forest with probabilities.
    device : str
        PyTorch device.
    return_tensors : bool
        If True, return ``dict[str, Tensor]`` instead of DataFrame.
        Keys: X, Y, Z, LAD, LAI, ReturnProbability, ReturnNumber.

    Returns
    -------
    DataFrame, dict[str, Tensor], or list
    """
    if torch is None:
        raise ImportError("PyTorch required. Install: uv pip install torch")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    # Parse input (DataFrame or tensor dict)
    X, Y, Z, LAD, _ = _parse_input(vxf, device)
    n = len(X)

    # Sort by X asc, Y asc, Z desc — must match pandas sort_values exactly
    # Use lexicographic sort: primary X, secondary Y, tertiary -Z
    neg_Z = -Z  # negate for descending Z
    # torch doesn't have lexsort, so convert to CPU numpy for sorting
    # (sorting is fast even on CPU, and ensures exact match with pandas)
    sort_idx_np = np.lexsort((neg_Z.cpu().numpy(), Y.cpu().numpy(), X.cpu().numpy()))
    sort_idx = torch.tensor(sort_idx_np, dtype=torch.int64, device=dev)
    X, Y, Z, LAD = X[sort_idx], Y[sort_idx], Z[sort_idx], LAD[sort_idx]
    seg = _encode_segments(X, Y)

    # Cumulative LAI
    LAI = _segment_cumsum(LAD, seg)

    # Return probability (Beer-Lambert)
    RP = torch.full((n,), float('nan'), dtype=torch.float64, device=dev)
    above = Z > 0
    ground = Z == 0
    RP[above] = P0_AGR * torch.exp(-k * LAI[above])
    RP[ground] = P0_GR * torch.exp(-k * LAI[ground])

    if LAD_prefactor:
        RP = RP * LAD

    # Random sampling (same seed as CPU)
    rng = np.random.default_rng(922)
    random_vec = rng.uniform(0, 1, size=n)
    random_t = torch.tensor(random_vec, dtype=torch.float64, device=dev)
    keep_mask = torch.isfinite(RP) & (RP > random_t)

    # Filter hits (stays on GPU)
    hit_X = X[keep_mask]
    hit_Y = Y[keep_mask]
    hit_Z = Z[keep_mask]
    hit_LAD = LAD[keep_mask]
    hit_LAI = LAI[keep_mask]
    hit_RP = RP[keep_mask]

    # Return numbering on GPU
    if hit_X.numel() > 0:
        hit_seg = _encode_segments(hit_X, hit_Y)
        hit_RN = _segment_cumcount(hit_seg)
    else:
        hit_RN = torch.zeros(0, dtype=torch.int64, device=dev)

    if return_tensors:
        hits = {
            "X": hit_X, "Y": hit_Y, "Z": hit_Z,
            "LAD": hit_LAD, "LAI": hit_LAI,
            "ReturnProbability": hit_RP, "ReturnNumber": hit_RN,
        }
        if prob_out:
            full = {
                "X": X, "Y": Y, "Z": Z,
                "LAD": LAD, "LAI": LAI, "ReturnProbability": RP,
            }
            return [hits, full]
        return hits

    # Convert to DataFrame
    hits_df = pd.DataFrame({
        "X": hit_X.cpu().numpy(),
        "Y": hit_Y.cpu().numpy(),
        "Z": hit_Z.cpu().numpy(),
        "LAD": hit_LAD.cpu().numpy(),
        "LAI": hit_LAI.cpu().numpy(),
        "ReturnProbability": hit_RP.cpu().numpy(),
        "ReturnNumber": hit_RN.cpu().numpy(),
    })

    if prob_out:
        full_df = pd.DataFrame({
            "X": X.cpu().numpy(), "Y": Y.cpu().numpy(), "Z": Z.cpu().numpy(),
            "LAD": LAD.cpu().numpy(), "LAI": LAI.cpu().numpy(),
            "ReturnProbability": RP.cpu().numpy(),
        })
        return [hits_df, full_df]

    return hits_df
