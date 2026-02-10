"""Tests for shared label-space utilities."""

import pytest

from src.label_space import build_label_context


def test_explicit_mode_is_trusted():
    checkpoint = {
        "model_config": {"num_grades": 3},
        "label_space_mode": "global_legacy",
        "grade_offset": 2,
        "min_grade_index": 2,
        "max_grade_index": 4,
    }
    context = build_label_context(checkpoint)
    assert context.label_space_mode == "global_legacy"
    assert context.global_to_model_label(2) == 2


def test_infers_remapped_for_compact_legacy_checkpoint():
    checkpoint = {
        "model_config": {"num_grades": 3},
        "grade_offset": 2,
        "min_grade_index": 2,
        "max_grade_index": 4,
    }
    context = build_label_context(checkpoint)
    assert context.label_space_mode == "remapped"
    assert context.global_to_model_label(2) == 0
    assert context.model_to_global_label(2) == 4


def test_infers_global_legacy_for_non_compact_legacy_checkpoint():
    checkpoint = {
        "model_config": {"num_grades": 19},
        "grade_offset": 2,
        "min_grade_index": 2,
        "max_grade_index": 4,
    }
    context = build_label_context(checkpoint)
    assert context.label_space_mode == "global_legacy"
    assert context.grade_offset == 0
    assert context.global_to_model_label(2) == 2


def test_single_grade_range_mapping():
    checkpoint = {
        "model_config": {"num_grades": 1},
        "label_space_mode": "remapped",
        "grade_offset": 2,
        "min_grade_index": 2,
        "max_grade_index": 2,
    }
    context = build_label_context(checkpoint)
    assert context.get_global_grade_bounds() == (2, 2)
    assert context.global_to_model_label(2) == 0
    assert context.model_to_global_label(0) == 2
    with pytest.raises(ValueError):
        context.global_to_model_label(3)
