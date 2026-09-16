import json
from pathlib import Path

import pytest

from edcr.config import load_config


def test_unknown_config_key_is_rejected(tmp_path):
    source = json.loads(Path("configs/pilot.yaml").read_text(encoding="utf-8"))
    source["surprise"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_default_config_is_valid():
    assert load_config("configs/pilot.yaml")["protocol_version"] == "v1"
