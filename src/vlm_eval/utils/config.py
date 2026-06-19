from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_config(path: str | Path) -> dict[str, object]:
    """Load a YAML configuration file and return its contents as a dict."""
    with open(path, encoding="utf-8") as handle:
        config: dict[str, object] = yaml.safe_load(handle)
    if config is None:
        return {}
    return config
