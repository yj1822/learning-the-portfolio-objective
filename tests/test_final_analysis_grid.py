from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_grid_module():
    path = PROJECT_ROOT / "scripts/run_final_analysis_grid.py"
    spec = importlib.util.spec_from_file_location("run_final_analysis_grid", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aum_baseline_filter_does_not_exclude_multiplier_ten() -> None:
    module = _load_grid_module()
    assert module._is_baseline_config(Path("aum_multiplier_1.yaml"), "aum")
    assert not module._is_baseline_config(Path("aum_multiplier_10.yaml"), "aum")


def test_frontier_baseline_filter_is_exact() -> None:
    module = _load_grid_module()
    assert module._is_baseline_config(Path("frontier_gamma_10.yaml"), "frontier")
    assert not module._is_baseline_config(Path("frontier_gamma_100.yaml"), "frontier")
