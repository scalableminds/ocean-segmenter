"""Build Neuroglancer viewer links for an output group (CLI convenience)."""

import json
import urllib.parse

from . import store

DEFAULT_VIEWER = "https://neuroglancer-demo.appspot.com/"
MAX_VISIBLE_SEGMENTS = 1000
DEPTH_SCALE = 50.0  # vertical exaggeration (relativeDisplayScales) of the elevation dimension


def state(
    base_url: str,
    layer: str,
    dims: list[str],
    shape: list[int],
    segment_ids: list[str] = (),
    source_layers: list[dict] = (),
    metadata: dict | None = None,
    depth_scale: float = DEPTH_SCALE,
) -> dict:
    """Viewer state showing the label array plus its meshes, in voxel index space.

    Neuroglancer interprets precomputed mesh vertices as nanometres, while our label dims are
    unitless voxel indices. The mesh source transform therefore scales by 1e9 (nm -> 1 voxel) and
    maps x -> last dim (longitude), y -> second to last (latitude), z -> third to last (vertical level).

    The vertical dimension is exaggerated by `depth_scale` via `relativeDisplayScales`, a display-only
    setting: labels and meshes are stretched together, coordinates and mesh data are unchanged.

    With `metadata` (the run's <layer>.metadata.json), the page title gets a short summary.
    """
    base_url = base_url.rstrip("/")
    spatial = dims[1:]
    mesh_axes = list(reversed(spatial))  # x, y, z -> lon, lat, depth
    out_dims = {d: [1, ""] for d in dims}
    sources: list[dict] = [{"url": f"zarr2://{base_url}/{layer}/"}]
    if len(spatial) == 3:
        sources.append(
            {
                "url": f"precomputed://{base_url}/meshes/{layer}",
                "transform": {
                    "outputDimensions": {d: [1, ""] for d in mesh_axes},
                    "inputDimensions": {d: [1e-9, "m"] for d in "xyz"},
                    "matrix": [[1e9, 0, 0, 0], [0, 1e9, 0, 0], [0, 0, 1e9, 0]],
                },
            }
        )
    position = [0.5] + [s / 2 for s in shape[1:]]
    extra = {"title": summary(metadata)} if metadata else {}
    return {
        **extra,
        "dimensions": out_dims,
        "displayDimensions": mesh_axes,
        **({"relativeDisplayScales": {spatial[0]: depth_scale}} if len(spatial) == 3 else {}),
        # Neuroglancer draws y downward; latitude increases northward. Rotate 180° about x -> north up.
        "crossSectionOrientation": [1, 0, 0, 0],
        "projectionOrientation": [1, 0, 0, 0],
        "position": position,
        "crossSectionScale": max(shape[1:]) / 500,
        "projectionScale": max(shape[1:]) * 1.5,
        "layers": [
            *source_layers,
            {
                "type": "segmentation",
                "name": layer,
                "source": sources,
                "segments": list(segment_ids)[:MAX_VISIBLE_SEGMENTS],
                "tab": "segments",
            },
        ],
        "layout": "xy-3d",
    }


def link(viewer_state: dict, viewer: str = DEFAULT_VIEWER) -> str:
    s = json.dumps(viewer_state, separators=(",", ":"))
    return f"{viewer.rstrip('/')}/#!{urllib.parse.quote(s, safe='')}"


def summary(metadata: dict) -> str:
    """Short human-readable description of a run, e.g. 'thetao in [28, inf] @ 2021-06-30'."""
    settings = metadata["settings"]
    rules = []
    for r in settings["rules"]:
        lo = "-inf" if r.get("min") is None else f"{r['min']:g}"
        hi = "inf" if r.get("max") is None else f"{r['max']:g}"
        rules.append(f"{r['variable']} in [{lo}, {hi}]")
    times = [t[:10] for t in settings["time_values"]]
    when = times[0] if len(times) == 1 else f"{times[0]}..{times[-1]}"
    joiner = f" {settings['op'].upper()} "
    return f"{joiner.join(rules)} @ {when}"


def load_metadata(output_url: str, layer: str) -> dict | None:
    """The run metadata written by `segment` (<layer>.metadata.json), if present."""
    return store.read_json(store.open_kv(output_url), f"{layer}.metadata.json")


def layer_meta(output_url: str, layer: str | None = None) -> tuple[str, list[str], list[int], list[str]]:
    """Layer name, dims, shape and segment ids (from the mesh segment properties) of an output group."""
    kv = store.open_kv(output_url)
    attrs = store.read_json(kv, ".zattrs") or {}
    layer = layer or attrs.get("multiscales", [{}])[-1].get("name", "segmentation")
    zarray = store.read_json(kv, f"{layer}/.zarray")
    zattrs = store.read_json(kv, f"{layer}/.zattrs") or {}
    if zarray is None:
        raise KeyError(f"No layer {layer!r} in {output_url}")
    props = store.read_json(kv, f"meshes/{layer}/segment_properties/info") or {}
    ids = props.get("inline", {}).get("ids", [])
    return layer, zattrs["_ARRAY_DIMENSIONS"], zarray["shape"], ids


_SHADER = """#uicontrol float lo slider(min={lo}, max={hi}, default={lo})
#uicontrol float hi slider(min={lo}, max={hi}, default={hi})
void main() {{
  int raw = getDataValue().value;
  if (raw == {fill}) {{ emitTransparent(); return; }}
  float v = float(raw) * {scale} + {offset};
  emitRGB(colormapJet(clamp((v - lo) / (hi - lo), 0.0, 1.0)));
}}
"""


def source_layers(metadata: dict) -> list[dict]:
    """Image layers for the source variable(s) a segmentation layer was derived from.

    The output holds only the selected time steps, so each source gets a translation along time
    that maps its time index onto the output's (contiguous) time axis. Values are decoded with
    scale_factor/add_offset in the shader (sliders in physical units); the fill value is transparent.
    Everything comes from the run metadata, so no access to the source is needed here.
    """
    group_url = metadata["input"]["group_url"]
    offset = metadata["settings"]["time_indices"][0]
    layers = []
    for name, var in metadata["input"]["variables"].items():
        attrs = var["attrs"]
        scale = float(attrs.get("scale_factor", 1.0))
        add = float(attrs.get("add_offset", 0.0))
        fill = var.get("fill_value")
        raw_lo, raw_hi = attrs.get("valid_min"), attrs.get("valid_max")
        if raw_lo is not None and raw_hi is not None:
            lo, hi = raw_lo * scale + add, raw_hi * scale + add
        else:
            rule = next(r for r in metadata["settings"]["rules"] if r["variable"] == name)
            lo, hi = rule.get("min") or 0.0, rule.get("max") or (rule.get("min") or 0.0) + 10
        rank = len(var["dims"])
        matrix = [[1 if c == r else 0 for c in range(rank)] + [-offset if r == 0 else 0] for r in range(rank)]
        layers.append(
            {
                "type": "image",
                "name": name,
                "source": {
                    "url": f"zarr2://{store.join(group_url, name)}/",
                    "transform": {"outputDimensions": {d: [1, ""] for d in var["dims"]}, "matrix": matrix},
                },
                "shader": _SHADER.format(
                    lo=f"{lo:.4g}", hi=f"{hi:.4g}", fill=int(fill) if isinstance(fill, (int, float)) else -2147483648,
                    scale=f"{scale:.9e}", offset=f"{add:.9e}",  # always valid GLSL float literals
                ),
                "visible": True,
            }
        )
    return layers
