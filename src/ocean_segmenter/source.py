"""Read CF-style Zarr v2 sources (e.g. CMEMS ARCO) with tensorstore."""

import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import tensorstore as ts

from . import store

TIME_DIM = "time"
_UNIT_SECONDS = {"days": 86400, "hours": 3600, "minutes": 60, "seconds": 1, "milliseconds": 1e-3}


@dataclass
class Variable:
    name: str
    dims: list[str]
    shape: list[int]
    zarray: dict[str, Any]
    attrs: dict[str, Any]
    array: ts.TensorStore

    def describe(self) -> dict[str, Any]:
        """Layout and attributes (incl. encoding: scale_factor, add_offset, fill) for the run metadata."""
        z = self.zarray
        return {
            "dims": self.dims,
            "shape": self.shape,
            "chunks": z.get("chunks"),
            "dtype": z.get("dtype"),
            "fill_value": z.get("fill_value"),
            "attrs": {k: v for k, v in self.attrs.items() if k != "_ARRAY_DIMENSIONS"},
        }

    def _read_raw(self, index: tuple) -> tuple[np.ndarray, np.ndarray]:
        """Read chunk by chunk along the remaining dims; inaccessible chunks (S3 403) count as missing.

        S3 buckets without list permission answer 403 instead of 404 for absent objects, which
        tensorstore reports as an error (the CMEMS ARCO stores do this, e.g. for the surface level).
        """
        view = self.array[index].translate_to[0]
        out = np.zeros(view.shape, view.dtype.numpy_dtype)
        denied = np.zeros(view.shape, bool)
        chunks = self.zarray["chunks"][len(index):]
        grids = [range(0, n, c) for n, c in zip(view.shape, chunks)]
        jobs = []
        for starts in np.ndindex(*[len(g) for g in grids]):
            local = tuple(slice(g[i], min(g[i] + c, n)) for g, i, c, n in zip(grids, starts, chunks, view.shape))
            jobs.append((local, view[local].read()))
        n_denied = 0
        for local, fut in jobs:
            try:
                out[local] = fut.result()
            except ValueError as e:
                if not store.is_access_denied(e):
                    raise
                denied[local] = True
                n_denied += 1
        if n_denied:
            warnings.warn(f"{self.name}{list(index)}: {n_denied}/{len(jobs)} chunks inaccessible (403), treated as missing")
        return out, denied

    def read(self, index: tuple = ()) -> np.ndarray:
        """Read and decode (scale_factor/add_offset, fill value -> NaN) to float32."""
        raw, missing = self._read_raw(index)
        data = raw.astype(np.float32)
        fill = self.zarray.get("fill_value")
        if fill is not None and not (isinstance(fill, str) or (isinstance(fill, float) and np.isnan(fill))):
            missing |= raw == fill
        if "missing_value" in self.attrs:
            missing |= raw == self.attrs["missing_value"]
        scale = self.attrs.get("scale_factor")
        offset = self.attrs.get("add_offset")
        if scale is not None:
            data *= np.float32(scale)
        if offset is not None:
            data += np.float32(offset)
        data[missing] = np.nan
        return data


class Source:
    """A Zarr v2 group. `url` may also point to one of its arrays; that array becomes `default_variable`."""

    def __init__(self, url: str, s3_endpoint: str | None = None):
        self.s3_endpoint = s3_endpoint
        self.default_variable: str | None = None
        kv = store.open_kv(url, s3_endpoint)
        if store.read_json(kv, ".zarray") is not None:
            parent, _, name = url.rstrip("/").rpartition("/")
            url, self.default_variable = parent, name
            kv = store.open_kv(url, s3_endpoint)
        self.url = url.rstrip("/")
        self.kv = kv
        consolidated = store.read_json(kv, ".zmetadata")
        self._meta: dict[str, Any] = consolidated["metadata"] if consolidated else {}
        self.attrs: dict[str, Any] = self._meta.get(".zattrs") or store.read_json(kv, ".zattrs") or {}

    def _array_meta(self, name: str) -> tuple[dict, dict]:
        zarray = self._meta.get(f"{name}/.zarray") or store.read_json(self.kv, f"{name}/.zarray")
        if zarray is None:
            raise KeyError(f"Variable {name!r} not found in {self.url}")
        zattrs = self._meta.get(f"{name}/.zattrs") or store.read_json(self.kv, f"{name}/.zattrs") or {}
        return zarray, zattrs

    def variables(self) -> list[str]:
        return sorted(k.removesuffix("/.zarray") for k in self._meta if k.endswith("/.zarray"))

    def variable(self, name: str) -> Variable:
        zarray, zattrs = self._array_meta(name)
        spec = {"driver": "zarr", "kvstore": store.kvstore_spec(store.join(self.url, name), self.s3_endpoint)}
        arr = ts.open(spec, read=True).result()
        dims = zattrs.get("_ARRAY_DIMENSIONS") or [f"dim_{i}" for i in range(len(zarray["shape"]))]
        return Variable(name, list(dims), list(zarray["shape"]), zarray, zattrs, arr)

    def coord(self, name: str) -> np.ndarray:
        return self.variable(name).array.read().result()

    def time_values(self) -> np.ndarray:
        return self.coord(TIME_DIM)

    def time_datetimes(self, values: np.ndarray) -> list[datetime]:
        units = self.variable(TIME_DIM).attrs.get("units", "")
        m = re.match(r"(\w+) since (.+)", units)
        if not m:
            raise ValueError(f"Cannot parse time units {units!r}")
        step = _UNIT_SECONDS[m.group(1)]
        epoch = datetime.fromisoformat(m.group(2).strip().replace("Z", "").replace(" ", "T"))
        epoch = epoch.replace(tzinfo=timezone.utc) if epoch.tzinfo is None else epoch
        return [epoch + timedelta(seconds=float(v) * step) for v in values]

    def resolve_time(self, time: str | int | None) -> list[int]:
        """Turn a time selector (index, ISO date, or 'start/end') into a list of time indices."""
        n = len(self.time_values())
        if time is None:
            return [0]
        if isinstance(time, int) or (isinstance(time, str) and time.lstrip("-").isdigit()):
            idx = int(time) % n
            return [idx]
        if "/" in time:
            start, end = time.split("/", 1)
            return list(range(self.resolve_time(start)[0], self.resolve_time(end)[0] + 1))
        target = datetime.fromisoformat(time)
        target = target.replace(tzinfo=timezone.utc) if target.tzinfo is None else target
        dts = self.time_datetimes(self.time_values())
        stamps = np.array([d.timestamp() for d in dts])
        idx = int(np.abs(stamps - target.timestamp()).argmin())
        if len(stamps) > 1 and abs(stamps[idx] - target.timestamp()) > np.median(np.diff(stamps)) / 2:
            warnings.warn(f"No time step at {time}; using nearest {dts[idx].isoformat()}")
        return [idx]
