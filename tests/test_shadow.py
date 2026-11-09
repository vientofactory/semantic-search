import pytest

from lawcast_semantic.shadow import REVISION, TEACHER, layer_positions, measure, validate_checkpoint


def test_shadow_layer_selection_preserves_final_blocks():
    layers = layer_positions(24)
    assert len(layers) == 20
    assert layers[-1] == 23
    with pytest.raises(ValueError, match='24 teacher layers'):
        layer_positions(12)


def test_shadow_rejects_an_incompatible_checkpoint():
    meta = dict(
        student='bge20', teacher=TEACHER, teacher_revision=REVISION, input_dim=1024, output=1024
    )
    validate_checkpoint(meta)
    with pytest.raises(ValueError, match='pinned teacher'):
        validate_checkpoint(dict(meta, teacher_revision='other'))


def test_shadow_timing_covers_full_pipeline_and_returns_serialized_value():
    timing, output = measure(lambda: 2, lambda x: x + 1, lambda x: x + 5)
    assert output == 8
    assert timing.total_ms == pytest.approx(
        timing.encode_ms + timing.index_ms + timing.serialize_ms
    )
