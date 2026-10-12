"""Mesh label volumes with serra and write Neuroglancer precomputed multi-resolution meshes.

Format: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md
(`neuroglancer_multilod_draco`, unsharded). The prototype writes a single LOD with a single
fragment per segment, so the fragment cell is simply the segment's bounding box.

Vertices are in voxel index space of the label array: x = last axis (longitude),
y = second to last (latitude), z = third to last (vertical level index, in source order), with voxel i
spanning [i, i+1).
"""

import struct
from concurrent.futures import Future

import DracoPy
import numpy as np
import serra_mesh
import tensorstore as ts

from . import store

QUANTIZATION_BITS = 16
_QMAX = (1 << QUANTIZATION_BITS) - 1


def info() -> dict:
    return {
        "@type": "neuroglancer_multilod_draco",
        "vertex_quantization_bits": QUANTIZATION_BITS,
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "lod_scale_multiplier": 1.0,
        "segment_properties": "segment_properties",
    }


def encode_segment(vertices: np.ndarray, faces: np.ndarray) -> tuple[bytes, bytes]:
    """Return (manifest `<id>.index`, fragment data `<id>`) for one mesh in voxel coordinates."""
    lo = vertices.min(0)
    extent = np.maximum(vertices.max(0) - lo, 1.0)
    quantized = np.rint((vertices - lo) / extent * _QMAX)
    # Neuroglancer reads the Draco integer positions without dequantizing, so the Draco lattice
    # must coincide with ours: range == 2**bits - 1 and origin == 0.
    fragment = DracoPy.encode(
        quantized,
        faces,
        quantization_bits=QUANTIZATION_BITS,
        quantization_range=_QMAX,
        quantization_origin=[0.0, 0.0, 0.0],
        compression_level=7,
    )
    manifest = b"".join(
        [
            np.asarray(extent, "<f4").tobytes(),  # chunk_shape
            np.asarray(lo, "<f4").tobytes(),  # grid_origin
            struct.pack("<I", 1),  # num_lods
            np.asarray([1.0], "<f4").tobytes(),  # lod_scales
            np.zeros(3, "<f4").tobytes(),  # vertex_offsets[lod]
            struct.pack("<I", 1),  # num_fragments_per_lod
            np.zeros(3, "<u4").tobytes(),  # fragment_positions (lod 0)
            struct.pack("<I", len(fragment)),  # fragment_offsets (sizes) (lod 0)
        ]
    )
    return manifest, fragment


def decode_segment(manifest: bytes, fragment: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `encode_segment` (single LOD/fragment), used by tests."""
    chunk_shape = np.frombuffer(manifest[0:12], "<f4")
    origin = np.frombuffer(manifest[12:24], "<f4")
    mesh = DracoPy.decode(fragment)
    vertices = origin + np.asarray(mesh.points, np.float64) / _QMAX * chunk_shape
    return vertices, np.asarray(mesh.faces).reshape(-1, 3)


class MeshWriter:
    def __init__(self, url: str, s3_endpoint: str | None = None):
        self.url = url.rstrip("/")
        self.kv = store.open_kv(self.url, s3_endpoint)
        self.kv.delete_range(ts.KvStore.KeyRange()).result()
        store.write_json(self.kv, "info", info())
        self.ids: list[int] = []
        self.labels: list[str] = []
        self.voxel_counts: list[int] = []
        self.time_indices: list[int] = []

    def mesh_volume(self, labels_zyx: np.ndarray, time_index: int, label: str, counts: dict[int, int]) -> None:
        """Mesh every non-zero label of a (z, y, x) volume and write its fragment + manifest."""
        if labels_zyx.ndim == 2:
            labels_zyx = labels_zyx[np.newaxis]
        mesher = serra_mesh.Mesher(voxel_resolution=[1, 1, 1], axis_order="ZYX")
        mesher.mesh(labels_zyx, close=True)
        pending: list[Future] = []
        for seg_id in mesher.ids():
            m = mesher.get(seg_id)
            mesher.erase(seg_id)
            if len(m.faces) == 0:
                continue
            # serra puts voxel centers at integer coordinates; Neuroglancer at i + 0.5.
            manifest, fragment = encode_segment(np.asarray(m.vertices, np.float64) + 0.5, m.faces)
            pending += [self.kv.write(f"{seg_id}.index", manifest), self.kv.write(f"{seg_id}", fragment)]
            self.ids.append(int(seg_id))
            self.labels.append(label)
            self.voxel_counts.append(int(counts.get(int(seg_id), 0)))
            self.time_indices.append(time_index)
            if len(pending) > 256:
                [f.result() for f in pending]
                pending.clear()
        [f.result() for f in pending]

    def finish(self) -> None:
        store.write_json(
            self.kv,
            "segment_properties/info",
            {
                "@type": "neuroglancer_segment_properties",
                "inline": {
                    "ids": [str(i) for i in self.ids],
                    "properties": [
                        {"id": "label", "type": "label", "values": self.labels},
                        {"id": "voxel_count", "type": "number", "data_type": "uint32", "values": self.voxel_counts},
                        {"id": "time_index", "type": "number", "data_type": "uint32", "values": self.time_indices},
                    ],
                },
            },
        )
