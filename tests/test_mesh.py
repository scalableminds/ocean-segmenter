import json

import numpy as np

from ocean_segmenter import store
from ocean_segmenter.mesh import MeshWriter, decode_segment


def test_cube_mesh_roundtrip(tmp_path):
    labels = np.zeros((10, 12, 14), np.uint32)  # z, y, x
    labels[2:5, 3:8, 4:12] = 7
    url = str(tmp_path / "meshes")
    w = MeshWriter(url)
    w.mesh_volume(labels, time_index=0, label="cube", counts={7: 120})
    w.finish()

    kv = store.open_kv(url)
    info = store.read_json(kv, "info")
    assert info["@type"] == "neuroglancer_multilod_draco"
    verts, faces = decode_segment(kv.read("7.index").result().value, kv.read("7").result().value)
    # voxel index space: x = last axis, voxel i spans [i, i+1)
    np.testing.assert_allclose(verts.min(0), [4, 3, 2], atol=1e-3)
    np.testing.assert_allclose(verts.max(0), [12, 8, 5], atol=1e-3)
    assert len(faces) > 0
    props = json.loads(kv.read("segment_properties/info").result().value)
    assert props["inline"]["ids"] == ["7"]
    assert props["inline"]["properties"][1]["values"] == [120]
