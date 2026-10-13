"""Patient-level 4-1-1 cross-validation protocol and dataset I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("image_2d", "rpm_1d", "no_surrogate")
PHASES = (50, 70, 90)
PATIENTS = tuple(range(1, 7))


def folds():
    """Six outer folds: test patient i, internal-validation patient i % 6 + 1, four training patients."""
    return [dict(fold=i, train=[p for p in PATIENTS if p not in (i, i % 6 + 1)],
                 validation=i % 6 + 1, test=i) for i in PATIENTS]


def validate_fold(split):
    train = split["train"]
    if len(train) != 4 or len(set(train) | {split["validation"], split["test"]}) != 6:
        raise ValueError("train/validation/test patients must be disjoint (4/1/1)")
    if set(train) | {split["validation"], split["test"]} != set(PATIENTS):
        raise ValueError("unexpected patient IDs")


def load_cases(root, patient_ids, forbidden=()):
    """Load ``patient_XX.npz`` files; refuse to load any patient listed in ``forbidden``."""
    ids = [int(x) for x in patient_ids]
    if set(ids) & set(forbidden):
        raise ValueError("attempted to load a held-out patient")
    cases = {}
    for patient in ids:
        with np.load(Path(root) / f"patient_{patient:02d}.npz", allow_pickle=False) as data:
            if int(data["patient"]) != patient:
                raise ValueError("patient ID differs from filename")
            cases[patient] = {name: data[name].copy() for name in ("target", "surrogate", "rpm")}
    return cases


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(data, patients):
    """Check cached inputs against ``manifest.json`` written by ``prepare_data``."""
    manifest = json.loads((Path(data) / "manifest.json").read_text())
    expected = {row["patient"]: row["sha256"] for row in manifest["patients"]}
    for patient in patients:
        if sha256(Path(data) / f"patient_{patient:02d}.npz") != expected[patient]:
            raise ValueError(f"input hash mismatch for patient {patient}")
    return manifest


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
