import hashlib
import json
from pathlib import Path


def test_recorded_artifact_hashes_match() -> None:
    manifest = json.loads(Path("research/ocr-small-models/hashes.json").read_text(encoding="utf-8"))
    assert manifest.pop("algorithm") == "sha256"
    for filename, expected in manifest.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
