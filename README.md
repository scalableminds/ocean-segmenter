# ocean-segmenter

Prototype for MyOcean. It thresholds ocean Zarr layers (for example CMEMS ARCO `so` and `thetao`) into **connected-component segmentation layers** and meshes each component for **Neuroglancer**.

```
source Zarr v2 ──► threshold (min/max per variable, AND/OR) ──► connected components (uint32, lon wraps)
                                                          ├──► <out>.zarr/<layer>            Zarr v2 labels (+ coords, CF/OME attrs)
                                                          └──► <out>.zarr/meshes/<layer>/    Neuroglancer multi-res (Draco) meshes
```

## Setup

```bash
uv sync
```

## CLI

The CLI flags are generated from the Pydantic request model (`SegmentRequest`); list/nested fields like `--rules` take JSON.

```bash
# Single variable: point at the array and give min/max
uv run ocean-seg segment \
  --source-url https://s3.waw3-1.cloudferro.com/mdl-arco-time-025/arco/GLOBAL_MULTIYEAR_PHY_001_030/cmems_mod_glo_phy_my_0.083deg_P1D-m_202311/timeChunked.zarr/so/ \
  --min 34.9 --max 35.1 --time 2021-06-30 --min-voxels 100 \
  --output-url out/so_band.zarr

# Multivariate rule (AND or OR over per-variable ranges)
uv run ocean-seg segment --source-url https://s3.waw3-1.cloudferro.com/mdl-arco-time-025/arco/GLOBAL_MULTIYEAR_PHY_001_030/cmems_mod_glo_phy_my_0.083deg_P1D-m_202311/timeChunked.zarr/ \
  --rules '[{"variable":"so","min":34.5,"max":35},{"variable":"thetao","min":2,"max":4}]' --op and \
  --time 2021-06-01/2021-06-07 --output-url s3://my-bucket/layers/antarctic_intermediate_water.zarr --s3-endpoint https://s3.waw3-1.cloudferro.com

# Regional dataset: Mediterranean Sea (4.2 km, 141 levels). Temperature is a separate
# dataset, so each request uses one variable. Here: the saltiest water (>= 39) on one summer day.
uv run ocean-seg segment \
  --source-url https://s3.waw3-1.cloudferro.com/mdl-arco-time-035/arco/MEDSEA_MULTIYEAR_PHY_006_004/cmems_mod_med_phy-sal_my_4.2km_P1D-m_202511/timeChunked.zarr/so/ \
  --min 39 --time 2021-08-15 --min-voxels 100 \
  --output-url out/med_salty.zarr

# Look at a result in Neuroglancer. Local outputs are served with CORS on http://127.0.0.1:<port>
# (--port, default 8765; a free port is picked if it is taken).
# The source variable(s) are added as image layers below the segmentation (--no-source to skip).
uv run ocean-seg view out/so_band.zarr
# ... and also print the Neuroglancer state as JSON (e.g. to paste into Neuroglancer's JSON editor)
uv run ocean-seg view out/so_band.zarr --print-json
```

To get one layer per variable, send one request per variable, for example with different `--layer-name`s into the same output group.

## Python API

The CLI is a thin wrapper around `run(SegmentRequest) -> SegmentResult`, so the same fields work from Python:

```python
from ocean_segmenter import SegmentRequest, ThresholdRule, run

req = SegmentRequest(
    source_url="https://s3.waw3-1.cloudferro.com/mdl-arco-time-025/arco/GLOBAL_MULTIYEAR_PHY_001_030/cmems_mod_glo_phy_my_0.083deg_P1D-m_202311/timeChunked.zarr/",
    rules=[ThresholdRule(variable="so", min=34.5, max=35), ThresholdRule(variable="thetao", min=2, max=4)],
    op="and",
    time="2021-06-01/2021-06-07",
    output_url="out/aaiw.zarr",
)

def on_progress(step: int, total: int, stage: str) -> None:
    print(f"{step}/{total} {stage}")  # stage: threshold | write | mesh | done
    # raise an exception here to cancel; it propagates out of run()

result = run(req, on_progress=on_progress)
print(result.num_segments, result.layer_path, result.mesh_path)
```

`run` is blocking. In an async app, call it with `await asyncio.to_thread(run, req)`. Progress is also logged to the `ocean_segmenter` logger.

## Request fields

| field | meaning |
|---|---|
| `source_url` | Zarr v2 group, or an array inside it (`…/timeChunked.zarr/so/`) |
| `rules` / `min`,`max` | per-variable inclusive ranges / shorthand for the variable in `source_url` |
| `op` | `and` / `or` to combine the rules |
| `time` | index, ISO date, or `start/end` (inclusive). Default is the first step. The spatial extent is always the full layer. |
| `connectivity` | 26 (default) or 6 (faces only) |
| `min_voxels` | drop components smaller than this |
| `mesh` | write meshes (default `true`) |
| `layer_name`, `output_url`, `s3_endpoint` | output location |

## Output layout

```
<out>.zarr/
  .zgroup .zattrs .zmetadata   provenance per layer + OME-NGFF 0.4 `multiscales` (consolidated, xarray-friendly)
  time/ elevation/ latitude/ longitude/   coordinates (selected time steps only)
  <layer>/                     uint32 labels, dims (time, elevation, latitude, longitude), gzip, source chunking
  meshes/<layer>/              neuroglancer_multilod_draco (1 LOD, 1 fragment per segment) + segment_properties
  <layer>.metadata.json        run record: input dataset (URL, dataset attrs, per-variable dims/units/encoding),
                               settings (rules, op, time selection + resolved indices/dates, connectivity,
                               min_voxels, mesh), output summary, tool version (also under .zattrs)
```

- Land and missing values (the source `_FillValue`) never pass a threshold.
- Components touching the antimeridian are merged, because longitude wraps.
- Segment IDs are unique across time steps, so a single mesh source covers the whole 4D layer.
- Mesh vertices are in **voxel index space**: x is longitude, y is latitude, and z is the vertical-level index in source order, with voxel *i* spanning [i, i+1). Real coordinates come from the coordinate arrays. In the CMEMS ARCO stores `elevation` is negative and deepest-first (index 0 is about -5728 m), and the levels are not evenly spaced.
- Chunks that answer **403** are treated as missing (fill value), as the Copernicus Marine toolbox does. These S3 buckets don't allow listing, so absent all-land chunks, such as most of the deepest level, return 403 instead of 404.
- The `view` link is built from `<layer>.metadata.json`; the page title is a short summary of the run (e.g. `thetao in [28, inf] @ 2021-06-30`). The full metadata stays in the output's `.zattrs` and `<layer>.metadata.json`.
- In the `view` link, each source variable is an image layer read straight from its Zarr store (CMEMS sends CORS headers). A time translation aligns it with the output's selected time steps, and the shader decodes `scale_factor`/`add_offset`, with `lo`/`hi` sliders in physical units and land made transparent.
- Neuroglancer treats mesh units as nm. The state from `ocean-seg view` therefore scales the mesh source by 1e9, so the meshes land on the voxel grid.

## Tests

```bash
uv run pytest
```

The tests use a synthetic store that mimics the CMEMS layout (int16 + scale/offset + fill).
