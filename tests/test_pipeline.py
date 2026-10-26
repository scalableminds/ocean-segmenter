import numpy as np
import pytest
import tensorstore as ts

from ocean_segmenter import pipeline, store
from ocean_segmenter.models import SegmentRequest, ThresholdRule
from ocean_segmenter.source import Source


def _labels(url, layer="segmentation"):
    return ts.open({"driver": "zarr", "kvstore": store.kvstore_spec(store.join(url, layer))}).result().read().result()


def test_source_decode_and_time(source):
    src = Source(source + "/so/")
    assert src.default_variable == "so"
    v = src.variable("so").read((0, 3))
    assert np.isnan(v[5]).all() and v[0, 0] == pytest.approx(30.0)
    assert src.resolve_time("2021-06-30") == [1]
    assert src.resolve_time("2021-06-29/2021-07-01") == [0, 1, 2]
    assert src.resolve_time(2) == [2]


def test_single_variable_shorthand(source, tmp_path):
    out = str(tmp_path / "out.zarr")
    res = pipeline.run(SegmentRequest(source_url=source + "/so/", min=34, max=36, time="0/1", output_url=out))
    assert res.shape == [2, 4, 6, 10]
    assert res.dims == ["time", "elevation", "latitude", "longitude"]
    # t=0: A + B (wrapped) ; t=1: single voxel + big blob
    assert res.num_segments == 4
    assert [s.time_index for s in res.segments] == [0, 0, 1, 1]
    labels = _labels(out)
    assert labels.dtype == np.uint32
    assert set(np.unique(labels[0])) == {0, 1, 2} and set(np.unique(labels[1])) == {0, 3, 4}
    assert labels[0, 1, 3, 0] == labels[0, 1, 3, 9]

    kv = store.open_kv(out)
    attrs = store.read_json(kv, ".zattrs")
    assert attrs["multiscales"][0]["datasets"][0]["path"] == "segmentation"
    assert "segmentation/.zarray" in store.read_json(kv, ".zmetadata")["metadata"]
    assert store.read_json(kv, "time/.zattrs")["units"] == "hours since 1950-01-01"
    for i in (1, 2, 3, 4):
        assert kv.read(f"meshes/segmentation/{i}.index").result().state == "value"


def test_min_voxels(source, tmp_path):
    res = pipeline.run(
        SegmentRequest(source_url=source + "/so/", min=34, time=1, min_voxels=2, output_url=str(tmp_path / "o.zarr"))
    )
    assert res.num_segments == 1 and res.segments[0].voxel_count == 18


@pytest.mark.parametrize("op,expected", [("and", 1), ("or", 2)])
def test_multivariate(source, tmp_path, op, expected):
    # t=0: thetao is cold (<=5) for x >= 5.
    # and: only the eastern part of blob B (x=8..9) -> 1
    # or:  the cold half merges with blob B through the lon wrap; blob A stays separate -> 2
    rules = [ThresholdRule(variable="so", min=34), ThresholdRule(variable="thetao", max=5)]
    out = str(tmp_path / "o.zarr")
    res = pipeline.run(SegmentRequest(source_url=source, rules=rules, op=op, mesh=False, output_url=out))
    assert res.num_segments == expected
    assert res.mesh_path is None
    assert (_labels(out)[0, 3, 5] == 0).all()  # land never segmented


def test_surface_broadcast(source, tmp_path):
    rules = [ThresholdRule(variable="so", min=34), ThresholdRule(variable="zos", min=0.4)]
    with pytest.warns(UserWarning, match="broadcast"):
        res = pipeline.run(SegmentRequest(source_url=source, rules=rules, mesh=False, output_url=str(tmp_path / "o.zarr")))
    assert res.num_segments == 1  # blob A (lat 1..2) is in the zos>0.4 band (lat 0..2)


def test_metadata_and_view_state(source, tmp_path):
    import json

    from ocean_segmenter import neuroglancer

    out = str(tmp_path / "o.zarr")
    rules = [ThresholdRule(variable="so", min=34), ThresholdRule(variable="thetao", max=5)]
    res = pipeline.run(SegmentRequest(source_url=source, rules=rules, time="2021-06-30/2021-07-01", output_url=out))

    # 1. metadata file next to the layer: input dataset + settings + output summary
    assert res.metadata_path.endswith("/segmentation.metadata.json")
    meta = neuroglancer.load_metadata(out, "segmentation")
    assert meta["input"]["group_url"] == source and meta["input"]["dataset_attrs"]["title"] == "synthetic"
    assert meta["input"]["variables"]["so"]["attrs"]["scale_factor"] == 0.01
    assert meta["input"]["variables"]["so"]["fill_value"] == -32767
    assert meta["settings"]["time_indices"] == [1, 2] and meta["settings"]["op"] == "and"
    assert meta["settings"]["rules"][1] == {"variable": "thetao", "min": None, "max": 5.0}
    assert meta["output"]["num_segments"] == res.num_segments
    assert store.read_json(store.open_kv(out), ".zattrs")["ocean_segmenter"]["segmentation"] == meta

    # source layers are built from the metadata alone
    layers = neuroglancer.source_layers(meta)
    assert [lyr["name"] for lyr in layers] == ["so", "thetao"]
    so = layers[0]
    assert so["source"]["url"] == f"zarr2://{source}/so/"
    assert so["source"]["transform"]["matrix"][0] == [1, 0, 0, 0, -1]  # output t=0 is source t=1
    assert "raw == -32767" in so["shader"] and "1.000000000e-02" in so["shader"]

    # 2. the viewer state title summarizes the run
    layer, dims, shape, ids = neuroglancer.layer_meta(out)
    state = neuroglancer.state("http://x", layer, dims, shape, ids, layers, meta)
    assert state["title"] == "so in [34, inf] AND thetao in [-inf, 5] @ 2021-06-30..2021-07-01"
    assert [lyr["type"] for lyr in state["layers"]] == ["image", "image", "segmentation"]


def test_public_api_progress_and_cancel(source, tmp_path):
    from ocean_segmenter import SegmentRequest as Req, run

    req = Req(source_url=source + "/so/", min=34, max=36, time="0/1", output_url=str(tmp_path / "o.zarr"))
    calls = []
    res = run(req, on_progress=lambda step, total, stage: calls.append((step, total, stage)))
    assert res.num_segments == 4
    assert calls == [
        (0, 2, "threshold"), (0, 2, "write"), (0, 2, "mesh"),
        (1, 2, "threshold"), (1, 2, "write"), (1, 2, "mesh"),
        (2, 2, "done"),
    ]

    class Cancelled(Exception):
        pass

    def cancel(step, total, stage):
        if step == 1:
            raise Cancelled

    with pytest.raises(Cancelled):
        run(req.model_copy(update={"output_url": str(tmp_path / "c.zarr")}), on_progress=cancel)
