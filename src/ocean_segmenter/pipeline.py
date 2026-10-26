"""End-to-end: source Zarr -> threshold -> connected components -> label Zarr (+ meshes)."""

import importlib.metadata
import logging
import warnings
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np

from . import segment, store
from .mesh import MeshWriter
from .models import SegmentInfo, SegmentRequest, SegmentResult, ThresholdRule
from .source import TIME_DIM, Source, Variable
from .writer import OutputGroup

log = logging.getLogger(__name__)

# on_progress(step, total, stage): step is the 0-based time step being processed, total the number of time
# steps, stage one of "threshold", "write", "mesh", or "done" (called once at the end with step == total).
# Raise from the callback to cancel the run; the exception propagates out of `run`.
ProgressCallback = Callable[[int, int, str], None]


def _resolve_rules(req: SegmentRequest, src: Source) -> list[ThresholdRule]:
    if req.rules:
        return req.rules
    if src.default_variable is None or (req.min is None and req.max is None):
        raise ValueError("Give `rules`, or point source_url at a variable and set `min`/`max`.")
    return [ThresholdRule(variable=src.default_variable, min=req.min, max=req.max)]


def _mask_time_step(
    variables: dict[str, Variable], rules: list[ThresholdRule], op: str, dims: list[str], t: int
) -> np.ndarray:
    """Combined boolean mask over the spatial dims of `dims` for time index t.

    Reads level by level along the first spatial axis to keep memory at one bool per voxel.
    Variables lacking that axis (e.g. surface fields) are broadcast along it.
    """
    spatial = dims[1:]
    shape = max(variables.values(), key=lambda v: len(v.dims)).shape[1:]
    if len(spatial) == 2:
        return segment.combine([segment.rule_mask(variables[r.variable].read((t,)), r) for r in rules], op)
    out = np.empty(shape, bool)
    cache_2d: dict[int, np.ndarray] = {}
    for k in range(shape[0]):
        masks = []
        for i, r in enumerate(rules):
            var = variables[r.variable]
            if var.dims[1:] == spatial:
                masks.append(segment.rule_mask(var.read((t, k)), r))
            else:
                if i not in cache_2d:
                    cache_2d[i] = segment.rule_mask(var.read((t,)), r)
                masks.append(cache_2d[i])
        out[k] = segment.combine(masks, op)
    return out


def run(req: SegmentRequest, on_progress: ProgressCallback | None = None) -> SegmentResult:
    progress = on_progress or (lambda step, total, stage: None)
    src = Source(req.source_url, req.s3_endpoint)
    rules = _resolve_rules(req, src)
    variables = {r.variable: src.variable(r.variable) for r in rules}

    ref = max(variables.values(), key=lambda v: len(v.dims))
    dims = ref.dims
    if dims[0] != TIME_DIM:
        raise ValueError(f"Expected {TIME_DIM!r} as first dimension of {ref.name}, got {dims}")
    for v in variables.values():
        if v.dims[0] != TIME_DIM or not set(v.dims) <= set(dims) or v.dims[-2:] != dims[-2:]:
            raise ValueError(f"Variable {v.name} {v.dims} is not compatible with {ref.name} {dims}")
        if v.dims != dims:
            warnings.warn(f"{v.name} {v.dims} is broadcast along {dims[1]!r} to match {ref.name}")

    time_indices = src.resolve_time(req.time)
    all_times = src.time_values()
    time_dts = src.time_datetimes(all_times[time_indices])
    layer = req.layer_name or "segmentation"
    rule_text = f" {req.op.upper()} ".join(r.describe() for r in rules)

    out = OutputGroup(req.output_url, req.s3_endpoint)
    time_var = src.variable(TIME_DIM)
    out.write_coord(TIME_DIM, all_times[time_indices], time_var.attrs)
    for d in dims[1:]:
        try:
            out.write_coord(d, src.coord(d), src.variable(d).attrs)
        except KeyError:
            log.warning("No coordinate array for dimension %s", d)

    shape = [len(time_indices), *ref.shape[1:]]
    chunks = [1, *ref.zarray["chunks"][1:]]
    labels_arr = out.create_labels(
        layer,
        dims,
        shape,
        chunks,
        {"long_name": f"Connected components of {rule_text}", "rules": [r.model_dump() for r in rules], "op": req.op},
    )
    mesh_url = store.join(req.output_url, "meshes", layer) if req.mesh else None
    meshes = MeshWriter(mesh_url, req.s3_endpoint) if mesh_url else None

    segments: list[SegmentInfo] = []
    offset = 0
    for i, (t, dt) in enumerate(zip(time_indices, time_dts)):
        log.info("time %s (%d/%d): thresholding", dt.date(), i + 1, len(time_indices))
        progress(i, len(time_indices), "threshold")
        mask = _mask_time_step(variables, rules, req.op, dims, t)
        labels, counts = segment.label(mask, req.connectivity, req.min_voxels, wrap_last_axis=True)
        del mask
        labels[labels > 0] += np.uint32(offset)
        ids = range(offset + 1, offset + len(counts) + 1)
        segments += [SegmentInfo(id=j, voxel_count=int(c), time_index=i) for j, c in zip(ids, counts)]
        log.info("time %s: %d segments, writing labels", dt.date(), len(counts))
        progress(i, len(time_indices), "write")
        labels_arr[i].write(labels).result()
        if meshes is not None and len(counts):
            log.info("time %s: meshing", dt.date())
            progress(i, len(time_indices), "mesh")
            meshes.mesh_volume(labels, i, f"{rule_text} @ {dt.date()}", dict(zip(ids, counts.tolist())))
        offset += len(counts)

    if meshes is not None:
        meshes.finish()
    metadata = {
        "layer": layer,
        "created": datetime.now(timezone.utc).isoformat(),
        "tool": {"name": "ocean-segmenter", "version": importlib.metadata.version("ocean-segmenter")},
        "input": {
            "source_url": req.source_url,
            "group_url": src.url,
            "s3_endpoint": req.s3_endpoint,
            "dataset_attrs": src.attrs,
            "variables": {name: v.describe() for name, v in variables.items()},
        },
        "settings": {
            "rules": [r.model_dump() for r in rules],
            "op": req.op,
            "time": req.time,
            "time_indices": time_indices,
            "time_values": [d.isoformat() for d in time_dts],
            "connectivity": req.connectivity,
            "min_voxels": req.min_voxels,
            "mesh": req.mesh,
        },
        "output": {
            "output_url": req.output_url,
            "labels": layer,
            "meshes": f"meshes/{layer}" if meshes else None,
            "dims": dims,
            "shape": shape,
            "num_segments": len(segments),
        },
    }
    metadata_key = f"{layer}.metadata.json"
    store.write_json(out.kv, metadata_key, metadata)
    out.update_attrs(layer, dims, metadata)
    out.consolidate()
    progress(len(time_indices), len(time_indices), "done")
    return SegmentResult(
        output_url=req.output_url,
        layer_path=store.join(req.output_url, layer),
        mesh_path=mesh_url,
        metadata_path=store.join(req.output_url, metadata_key),
        dims=dims,
        shape=shape,
        time_values=[d.isoformat() for d in time_dts],
        rules=rules,
        op=req.op,
        num_segments=len(segments),
        segments=segments,
    )
