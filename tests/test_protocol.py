import pytest

from demt_li.protocol import folds, validate_fold


def test_six_disjoint_patient_level_folds():
    splits = folds()
    assert [s["test"] for s in splits] == [1, 2, 3, 4, 5, 6]
    assert [s["validation"] for s in splits] == [2, 3, 4, 5, 6, 1]
    for split in splits:
        validate_fold(split)
        assert len(split["train"]) == 4
        assert {split["test"], split["validation"]}.isdisjoint(split["train"])


def test_overlapping_split_is_rejected():
    with pytest.raises(ValueError):
        validate_fold(dict(train=[1, 2, 3, 4], validation=4, test=5))
