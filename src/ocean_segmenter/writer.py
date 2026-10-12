"""Write the output Zarr v2 group: label array, copied coordinates, CF + OME-NGFF attributes."""

import json
from typing import Any

import numpy as np
import tensorstore as ts

from . import store

# gzip: decodable everywhere (Neuroglancer decodes it natively, blosc/zstd need its WASM codecs).
COMPRESSOR = {"id": "gzip", "level": 5}


def _create_array(url: str, s3_endpoint: str | None, shape, chunks, dtype: np.dtype, fill_value) -> ts.TensorStore:
    spec = {
        "driver": "zarr",
        "kvstore": store.kvstore_spec(url, s3_endpoint),
        "metadata": {
            "shape": list(shape),
            "chunks": list(chunks),
            "dtype": np.dtype(dtype).str,
            "compressor": COMPRESSOR,
            "fill_value": fill_value,
            "order": "C",
        },
        "create": True,
        "delete_existing": True,
    }
    return ts.open(spec).result()


class OutputGroup:
    def __init__(self, url: str, s3_endpoint: str | None = None):
        self.url = url.rstrip("/")
        self.s3_endpoint = s3_endpoint
        self.kv = store.open_kv(self.url, s3_endpoint)
        store.write_json(self.kv, ".zgroup", {"zarr_format": 2})

    def clear_prefix(self, prefix: str) -> None:
        prefix = prefix.rstrip("/") + "/"
        self.kv.delete_range(ts.KvStore.KeyRange(prefix, prefix[:-1] + "0")).result()

    def write_coord(self, name: str, values: np.ndarray, attrs: dict[str, Any]) -> None:
        arr = _create_array(
            store.join(self.url, name), self.s3_endpoint, values.shape, values.shape, values.dtype, None
        )
        arr.write(values).result()
        store.write_json(self.kv, f"{name}/.zattrs", {**attrs, "_ARRAY_DIMENSIONS": [name]})

    def create_labels(
        self, name: str, dims: list[str], shape: list[int], chunks: list[int], attrs: dict[str, Any]
    ) -> ts.TensorStore:
        self.clear_prefix(name)
        arr = _create_array(store.join(self.url, name), self.s3_endpoint, shape, chunks, np.uint32, 0)
        store.write_json(self.kv, f"{name}/.zattrs", {**attrs, "_ARRAY_DIMENSIONS": dims})
        return arr

    def update_attrs(self, layer: str, dims: list[str], provenance: dict[str, Any]) -> None:
        """Merge group attrs: per-layer provenance plus an OME-NGFF 0.4 multiscales entry (read by Neuroglancer)."""
        attrs = store.read_json(self.kv, ".zattrs") or {}
        axes = [{"name": d, "type": "time" if d == "time" else "space"} for d in dims]
        entry = {
            "version": "0.4",
            "name": layer,
            "axes": axes,
            "datasets": [{"path": layer, "coordinateTransformations": [{"type": "scale", "scale": [1.0] * len(dims)}]}],
        }
        attrs["multiscales"] = [m for m in attrs.get("multiscales", []) if m.get("name") != layer] + [entry]
        attrs.setdefault("ocean_segmenter", {})[layer] = provenance
        store.write_json(self.kv, ".zattrs", attrs)

    def consolidate(self) -> None:
        """Write .zmetadata so xarray-style consumers can open the group with consolidated=True."""
        meta: dict[str, Any] = {}
        for key in self.kv.list().result():
            k = key.decode()
            if k.rsplit("/", 1)[-1] in (".zarray", ".zattrs", ".zgroup"):
                meta[k] = json.loads(self.kv.read(k).result().value)
        store.write_json(self.kv, ".zmetadata", {"zarr_consolidated_format": 1, "metadata": meta})
