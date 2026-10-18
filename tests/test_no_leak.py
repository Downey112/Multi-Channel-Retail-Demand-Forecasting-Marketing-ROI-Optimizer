"""The estimators must not see the simulator's truth: no import of mcr.simulate, no true_params.json."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_FILES = [ROOT / "src" / "mcr" / "analysis.py", ROOT / "src" / "mcr" / "transforms.py",
                  ROOT / "scripts" / "03_experiment.py", ROOT / "scripts" / "04_response.py",
                  ROOT / "scripts" / "05_forecast.py", ROOT / "app.py"]


@pytest.mark.parametrize("path", ANALYSIS_FILES, ids=lambda p: p.name)
def test_analysis_code_never_touches_the_truth(path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").endswith("simulate"), f"{path.name} imports {node.module}"
            assert all(a.name != "simulate" for a in node.names), f"{path.name} imports simulate"
        if isinstance(node, ast.Import):
            assert all(not a.name.endswith("simulate") for a in node.names)
    assert "true_params" not in src and "TRUE[" not in src
