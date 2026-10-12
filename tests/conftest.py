"""Synthetic CMEMS-like Zarr v2 source: int16 + scale/offset + fill, consolidated metadata."""

import numpy as np
import pytest
import tensorstore as ts

from ocean_segmenter import store

FILL = -32767
SCALE, OFFSET = 0.01, 30.0
NT, NZ, NY, NX = 3, 4, 6, 10


def _write(url: str, name: str, data: np.ndarray, chunks, attrs: dict, fill=None) -> None:
    arr = ts.open(
        {
            "driver": "zarr",
            "kvstore": store.kvstore_spec(store.join(url, name)),
            "metadata": {"shape": list(data.shape), "chunks": list(chunks), "dtype": data.dtype.str, "fill_value": fill},
            "create": True,
        }
    ).result()
    arr.write(data).result()
    store.write_json(store.open_kv(url), f"{name}/.zattrs", attrs)


def encode(values: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(values), FILL, np.rint((np.nan_to_num(values) - OFFSET) / SCALE)).astype(np.int16)


def make_source(url: str, so: np.ndarray, thetao: np.ndarray | None = None, zos: np.ndarray | None = None) -> None:
    kv = store.open_kv(url)
    store.write_json(kv, ".zgroup", {"zarr_format": 2})
    store.write_json(kv, ".zattrs", {"title": "synthetic"})
    hours = (np.arange(NT, dtype=np.float32) * 24) + 26112 * 24  # 2021-06-29 onwards
    _write(url, "time", hours, [NT], {"_ARRAY_DIMENSIONS": ["time"], "units": "hours since 1950-01-01"})
    _write(url, "elevation", np.array([0.5, 10, 50, 200], np.float32), [NZ],
           {"_ARRAY_DIMENSIONS": ["elevation"], "units": "m", "positive": "down"})
    _write(url, "latitude", np.linspace(-10, 10, NY, dtype=np.float32), [NY], {"_ARRAY_DIMENSIONS": ["latitude"]})
    _write(url, "longitude", np.linspace(-180, 144, NX, dtype=np.float32), [NX], {"_ARRAY_DIMENSIONS": ["longitude"]})
    var_attrs = {"scale_factor": SCALE, "add_offset": OFFSET}
    dims4 = ["time", "elevation", "latitude", "longitude"]
    _write(url, "so", encode(so), [1, 1, NY, NX], {**var_attrs, "_ARRAY_DIMENSIONS": dims4}, FILL)
    if thetao is not None:
        _write(url, "thetao", encode(thetao), [1, 1, NY, NX], {**var_attrs, "_ARRAY_DIMENSIONS": dims4}, FILL)
    if zos is not None:
        _write(url, "zos", encode(zos), [1, NY, NX],
               {**var_attrs, "_ARRAY_DIMENSIONS": ["time", "latitude", "longitude"]}, FILL)
    # consolidate like the ARCO stores
    from ocean_segmenter.writer import OutputGroup

    OutputGroup(url).consolidate()


@pytest.fixture
def blobs():
    """so field: background 30, two separate blobs of 35 at t=0 (one wraps around longitude), land as NaN."""
    so = np.full((NT, NZ, NY, NX), 30.0)
    so[0, 0:2, 0:2, 1:3] = 35.0  # blob A
    so[0, 1:3, 3:5, 0:2] = 35.0  # blob B, touches x=0 ...
    so[0, 1:3, 3:5, 8:10] = 35.0  # ... and x=NX-1 -> wraps into B
    so[1, 0, 0, 5] = 35.0  # single-voxel blob at t=1
    so[1, 2:4, 2:5, 4:7] = 35.0  # bigger blob at t=1
    so[:, 3, 5, :] = np.nan  # land
    return so


@pytest.fixture
def source(tmp_path, blobs):
    url = str(tmp_path / "src.zarr")
    thetao = np.full_like(blobs, 10.0)
    thetao[0, :, :, 5:] = 3.0  # cold on the eastern half
    thetao[np.isnan(blobs)] = np.nan
    zos = np.zeros((NT, NY, NX))
    zos[:, :3] = 0.5
    make_source(url, blobs, thetao, zos)
    return url
