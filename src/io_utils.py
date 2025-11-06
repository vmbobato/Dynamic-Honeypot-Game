import os, json
from pathlib import Path


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def write_json(path: str, obj) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)