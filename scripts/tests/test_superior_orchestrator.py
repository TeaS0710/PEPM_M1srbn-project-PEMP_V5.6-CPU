import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.superior.superior_orchestrator import generate_run_plan, load_exp_config


def make_minimal_config(tmp_path: Path) -> Path:
    cfg = {
        "exp_id": "unit_exp",
        "description": "Minimal exp for tests",
        "base": {
            "profile": "demo_profile",
            "stage": "pipeline",
            "fixed": {"CORPUS_ID": "web1"},
            "overrides": {"ideology.view": "global"},
        },
        "axes": [
            {
                "name": "dataset",
                "type": "choice",
                "values": [
                    {"label": "a", "overrides": {"data.corpus_ids": ["web1"]}},
                    {"label": "b", "overrides": {"data.corpus_ids": ["web2"]}},
                ],
            }
        ],
        "grid": {"mode": "cartesian"},
        "run": {"repeats": 1, "seed_strategy": "fixed", "base_seed": 7},
        "scheduler": {"parallel": 1, "max_weight": 2, "resource_classes": {"sklearn": "light"}},
    }
    cfg_path = tmp_path / "exp.yml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_path


def test_load_exp_config(tmp_path: Path):
    cfg_path = make_minimal_config(tmp_path)
    exp_cfg = load_exp_config(str(cfg_path))

    assert exp_cfg.exp_id == "unit_exp"
    assert exp_cfg.base_profile == "demo_profile"
    assert exp_cfg.base_make_vars["CORPUS_ID"] == "web1"
    assert exp_cfg.axes[0].name == "dataset"
    assert exp_cfg.scheduler.parallel == 1


def test_generate_run_plan(tmp_path: Path):
    cfg_path = make_minimal_config(tmp_path)
    exp_cfg = load_exp_config(str(cfg_path))
    plan = generate_run_plan(exp_cfg)

    assert len(plan) == 2  # two axis values x 1 repeat
    assert plan[0].run_id.endswith("000000")
    assert plan[1].axis_values["dataset"] == "b"
    assert plan[0].make_vars["SEED"] == "7"


def test_dry_run_writes_plan(tmp_path: Path):
    cfg_path = make_minimal_config(tmp_path)
    exp_cfg = load_exp_config(str(cfg_path))
    exp_dir = Path("superior") / exp_cfg.exp_id
    if exp_dir.exists():
        for child in exp_dir.glob("**/*"):
            if child.is_file():
                child.unlink()
        for child in sorted(exp_dir.glob("**/*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        exp_dir.rmdir()

    proc = subprocess.run(
        [
            "python",
            "-m",
            "scripts.superior.superior_orchestrator",
            "--exp-config",
            str(cfg_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan_path = exp_dir / "plan.tsv"
    assert plan_path.exists()
    content = plan_path.read_text(encoding="utf-8")
    assert "run_id" in content.splitlines()[0]
    assert f"Generated {len(generate_run_plan(exp_cfg))} runs" in proc.stdout
