import numpy as np

from ocean_segmenter import segment
from ocean_segmenter.models import ThresholdRule


def test_rule_mask_inclusive_and_nan():
    v = np.array([1.0, 2.0, 3.0, np.nan])
    assert segment.rule_mask(v, ThresholdRule(variable="x", min=2, max=3)).tolist() == [False, True, True, False]
    assert segment.rule_mask(v, ThresholdRule(variable="x", max=1)).tolist() == [True, False, False, False]
    assert segment.rule_mask(v, ThresholdRule(variable="x")).tolist() == [True, True, True, False]


def test_combine_and_or():
    a, b = np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool)
    assert segment.combine([a, b], "and").tolist() == [True, False, False, False]
    assert segment.combine([a, b], "or").tolist() == [True, True, True, False]


def test_label_two_blobs_and_wrap():
    m = np.zeros((2, 4, 10), bool)
    m[0, 0:2, 1:3] = True
    m[1, 2:4, 0] = True
    m[1, 2:4, 9] = True
    labels, counts = segment.label(m, wrap_last_axis=False)
    assert labels.max() == 3
    labels, counts = segment.label(m, wrap_last_axis=True)
    assert labels.max() == 2 and labels.dtype == np.uint32
    assert labels[1, 2, 0] == labels[1, 2, 9]
    assert sorted(counts.tolist()) == [4, 4]


def test_label_min_voxels_renumbers():
    m = np.zeros((1, 5, 5), bool)
    m[0, 0, 0] = True
    m[0, 2:5, 2:5] = True
    labels, counts = segment.label(m, min_voxels=2)
    assert labels.max() == 1 and counts.tolist() == [9]
    assert labels[0, 0, 0] == 0


def test_label_2d():
    m = np.zeros((4, 4), bool)
    m[0, 0] = m[1, 1] = True
    assert segment.label(m, connectivity=6)[0].max() == 2
    assert segment.label(m, connectivity=26)[0].max() == 1
