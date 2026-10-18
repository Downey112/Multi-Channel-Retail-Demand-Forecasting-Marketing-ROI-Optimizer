"""Result files under docs/results (JSON) that the README, the app and the recovery check read."""
import json

import numpy as np

from .db import ROOT

RESULTS = ROOT / "docs" / "results"


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save(name: str, obj: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")
    print(f"  results -> docs/results/{name}.json")


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
