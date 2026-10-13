"""Load ``configs/config.yaml``."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@lru_cache(maxsize=None)
def load_config(path=CONFIG_PATH):
    with Path(path).open() as stream:
        return yaml.safe_load(stream)
