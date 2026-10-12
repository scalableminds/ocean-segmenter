"""Thresholding and connected-component labeling of one time step."""

from functools import reduce

import cc3d
import numpy as np

from .models import ThresholdRule


def rule_mask(values: np.ndarray, rule: ThresholdRule) -> np.ndarray:
    """Inclusive range test; NaN (land / missing) is never inside."""
    mask = ~np.isnan(values)
    if rule.min is not None:
        mask &= values >= rule.min
    if rule.max is not None:
        mask &= values <= rule.max
    return mask


def combine(masks: list[np.ndarray], op: str) -> np.ndarray:
    return reduce(np.logical_and if op == "and" else np.logical_or, masks)


def _merge_wrap(labels: np.ndarray, n: int) -> None:
    """Merge components touching across the last/first index of the last (longitude) axis. In place."""
    first, last = labels[..., 0].ravel(), labels[..., -1].ravel()
    touching = (first > 0) & (last > 0)
    if not touching.any():
        return
    parent = np.arange(n + 1, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = np.unique(np.stack([first[touching], last[touching]], 1), axis=0)
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.arange(n + 1, dtype=labels.dtype)
    for x in np.unique(pairs):
        roots[x] = find(x)
    labels[...] = roots[labels]


def label(
    mask: np.ndarray, connectivity: int = 6, min_voxels: int = 0, wrap_last_axis: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of `mask` (2D or 3D).

    Returns uint32 labels numbered 1..N consecutively and the voxel count of each label
    (`counts[i]` is the size of label i+1).
    """
    if mask.ndim == 2:
        connectivity = 4 if connectivity == 6 else 8
    labels, n = cc3d.connected_components(mask, connectivity=connectivity, return_N=True, out_dtype=np.uint32)
    if wrap_last_axis and n:
        _merge_wrap(labels, n)
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    counts[0] = 0
    keep = counts >= max(min_voxels, 1)
    remap = np.zeros(n + 1, dtype=np.uint32)
    remap[keep] = np.arange(1, keep.sum() + 1, dtype=np.uint32)
    labels = remap[labels]
    return labels, counts[keep]
