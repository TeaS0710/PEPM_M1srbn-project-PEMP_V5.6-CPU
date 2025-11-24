"""Superior orchestration layer (V5).

This module parses experiment configs, generates run plans, and schedules
runs that delegate execution to the existing core through ``make run``
(via :mod:`scripts.superior.run_single`). The implementation follows the
V5 specification in ``dev_V5.md`` while staying pragmatic and backward
compatible with the V4 core.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from scripts.core.core_utils import load_yaml


# ---------------------------------------------------------------------------
# Data classes (see dev_V5.md §4)
# ---------------------------------------------------------------------------
@dataclass
class AxisValueConfig:
    label: str
    make_vars: Dict[str, str]
    overrides: Dict[str, Any]


@dataclass
class AxisConfig:
    name: str
    type: str  # "choice" (plus tard "manual")
    values: List[AxisValueConfig]


@dataclass
class SchedulerConfig:
    parallel: int
    max_ram_gb: Optional[float]
    resource_classes: Dict[str, str]
    weights: Dict[str, int]
    max_weight: int


@dataclass
class AnalysisHooksConfig:
    after_experiment: List[Dict[str, Any]]


@dataclass
class ExpConfig:
    exp_id: str
    description: str
    base_profile: str
    base_stage: str
    base_make_vars: Dict[str, str]
    base_overrides: Dict[str, Any]
    axes: List[AxisConfig]
    grid_mode: str
    repeats: int
    seed_strategy: str
    base_seed: int
    scheduler: SchedulerConfig
    analysis_hooks: AnalysisHooksConfig


@dataclass
class RunSpec:
    run_id: str
    exp_id: str
    profile: str
    stage: str
    make_vars: Dict[str, str]
    overrides: Dict[str, Any]
    repeat_index: int
    axis_values: Dict[str, str]
    resource_class: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {"light": 1, "medium": 2, "heavy": 4}


def _flatten_overrides(prefix: str, value: Any) -> Dict[str, Any]:
    """Flatten nested overrides using dotted paths.

    Example::
        {"a": {"b": 1}} -> {"a.b": 1}
    """
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for sub_key, sub_val in value.items():
            new_prefix = f"{prefix}.{sub_key}" if prefix else sub_key
            result.update(_flatten_overrides(new_prefix, sub_val))
        return result
    return {prefix: value}


def _merge_dicts(dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for d in dicts:
        merged.update(d or {})
    return merged


def _format_override_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _resource_class_for(make_vars: Dict[str, str], scheduler: SchedulerConfig) -> str:
    family = make_vars.get("FAMILY") or make_vars.get("family")
    if family:
        normalized_family = str(family)
        if normalized_family in scheduler.resource_classes:
            return scheduler.resource_classes[normalized_family]
        lower_family = normalized_family.lower()
        if lower_family in scheduler.resource_classes:
            return scheduler.resource_classes[lower_family]
    return "light"


# ---------------------------------------------------------------------------
# Loading / parsing experiment configuration
# ---------------------------------------------------------------------------

def load_exp_config(path: str) -> ExpConfig:
    """Load an experiment configuration YAML into typed structures."""
    raw = load_yaml(path)

    base_cfg = raw.get("base") or {}
    axes_cfg = raw.get("axes") or []
    grid_cfg = raw.get("grid") or {}
    run_cfg = raw.get("run") or {}
    scheduler_cfg = raw.get("scheduler") or {}
    hooks_cfg = raw.get("analysis_hooks") or {}

    axes: List[AxisConfig] = []
    for axis_raw in axes_cfg:
        values_raw = axis_raw.get("values") or []
        values: List[AxisValueConfig] = []
        for val_raw in values_raw:
            values.append(
                AxisValueConfig(
                    label=val_raw.get("label", ""),
                    make_vars=val_raw.get("make_vars") or {},
                    overrides=val_raw.get("overrides") or {},
                )
            )
        axes.append(
            AxisConfig(
                name=axis_raw.get("name", "axis"),
                type=axis_raw.get("type", "choice"),
                values=values,
            )
        )

    scheduler = SchedulerConfig(
        parallel=int(scheduler_cfg.get("parallel", 1)),
        max_ram_gb=scheduler_cfg.get("max_ram_gb"),
        resource_classes=scheduler_cfg.get("resource_classes") or {},
        weights={**DEFAULT_WEIGHTS, **(scheduler_cfg.get("weights") or {})},
        max_weight=int(scheduler_cfg.get("max_weight", 4)),
    )

    analysis_hooks = AnalysisHooksConfig(
        after_experiment=hooks_cfg.get("after_experiment") or [],
    )

    return ExpConfig(
        exp_id=raw.get("exp_id") or Path(path).stem,
        description=raw.get("description", ""),
        base_profile=base_cfg.get("profile", ""),
        base_stage=base_cfg.get("stage", "pipeline"),
        base_make_vars=base_cfg.get("fixed") or {},
        base_overrides=base_cfg.get("overrides") or {},
        axes=axes,
        grid_mode=grid_cfg.get("mode", "cartesian"),
        repeats=int(run_cfg.get("repeats", 1)),
        seed_strategy=run_cfg.get("seed_strategy", "fixed"),
        base_seed=int(run_cfg.get("base_seed", 42)),
        scheduler=scheduler,
        analysis_hooks=analysis_hooks,
    )


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

def generate_run_plan(exp_config: ExpConfig) -> List[RunSpec]:
    """Expand an experiment configuration into a list of runs."""
    if exp_config.grid_mode != "cartesian":
        raise ValueError(f"Unsupported grid mode: {exp_config.grid_mode}")

    axes_values: List[List[AxisValueConfig]] = [axis.values for axis in exp_config.axes]
    if not axes_values:
        axes_values = [[]]  # single combination with no axes

    plan: List[RunSpec] = []
    combo_iterator: Iterable[Sequence[AxisValueConfig]] = itertools.product(*axes_values)
    counter = 0

    for combo in combo_iterator:
        axis_values_map: Dict[str, str] = {}
        combo_make_vars: List[Dict[str, str]] = []
        combo_overrides: List[Dict[str, Any]] = []

        for axis, axis_val in zip(exp_config.axes, combo):
            axis_values_map[axis.name] = axis_val.label
            combo_make_vars.append(axis_val.make_vars)
            combo_overrides.append(axis_val.overrides)

        base_make_vars = {k: str(v) for k, v in (exp_config.base_make_vars or {}).items()}
        combined_make_vars: Dict[str, str] = _merge_dicts([base_make_vars] + combo_make_vars)
        combined_overrides: Dict[str, Any] = _merge_dicts(
            [exp_config.base_overrides] + combo_overrides
        )

        for repeat_idx in range(exp_config.repeats):
            run_id = f"{exp_config.exp_id}_run_{counter:06d}"
            seed_value = exp_config.base_seed
            if exp_config.seed_strategy == "per_run":
                seed_value = exp_config.base_seed + counter
            combined_make_vars_with_seed = dict(combined_make_vars)
            combined_make_vars_with_seed.setdefault("SEED", str(seed_value))

            resource_class = _resource_class_for(
                combined_make_vars_with_seed, exp_config.scheduler
            )

            plan.append(
                RunSpec(
                    run_id=run_id,
                    exp_id=exp_config.exp_id,
                    profile=exp_config.base_profile,
                    stage=exp_config.base_stage,
                    make_vars=combined_make_vars_with_seed,
                    overrides=combined_overrides,
                    repeat_index=repeat_idx,
                    axis_values=axis_values_map,
                    resource_class=resource_class,
                )
            )
            counter += 1

    return plan


# ---------------------------------------------------------------------------
# Plan / run persistence (TSV helpers)
# ---------------------------------------------------------------------------
PLAN_COLUMNS = [
    "run_id",
    "exp_id",
    "profile",
    "stage",
    "repeat_index",
    "axis_values_json",
    "make_vars_json",
    "overrides_json",
    "resource_class",
]


RUN_COLUMNS = [
    "run_id",
    "exp_id",
    "profile",
    "stage",
    "status",
    "return_code",
    "family",
    "model_id",
    "corpus_id",
    "dataset_id",
    "view",
    "axis_values_json",
    "make_vars_json",
    "overrides_json",
    "metrics_path",
    "log_path",
    "started_at",
    "finished_at",
    "duration_s",
]


def _write_tsv(rows: List[Dict[str, Any]], path: Path, columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_plan_tsv(plan: List[RunSpec], path: Path) -> None:
    rows = []
    for run in plan:
        rows.append(
            {
                "run_id": run.run_id,
                "exp_id": run.exp_id,
                "profile": run.profile,
                "stage": run.stage,
                "repeat_index": run.repeat_index,
                "axis_values_json": json.dumps(run.axis_values, ensure_ascii=False),
                "make_vars_json": json.dumps(run.make_vars, ensure_ascii=False),
                "overrides_json": json.dumps(run.overrides, ensure_ascii=False),
                "resource_class": run.resource_class,
            }
        )
    _write_tsv(rows, path, PLAN_COLUMNS)


def read_runs_tsv(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows[row["run_id"]] = row
    return rows


def write_runs_tsv(rows: Dict[str, Dict[str, Any]], path: Path) -> None:
    ordered_rows = [rows[k] for k in sorted(rows.keys())]
    _write_tsv(ordered_rows, path, RUN_COLUMNS)


# ---------------------------------------------------------------------------
# Scheduler helpers
# ---------------------------------------------------------------------------

def _build_override_items(overrides: Dict[str, Any]) -> List[str]:
    flat: Dict[str, Any] = {}
    for key, value in overrides.items():
        flat.update(_flatten_overrides(key, value))
    return [f"{k}={_format_override_value(v)}" for k, v in flat.items()]


def _infer_metadata(make_vars: Dict[str, str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    dataset_id = (
        overrides.get("dataset_id")
        or make_vars.get("DATASET_ID")
        or make_vars.get("CORPUS_ID")
    )
    view = overrides.get("ideology.view") or overrides.get("view")
    family = make_vars.get("FAMILY") or ""
    model_id = make_vars.get("MODEL_ID") or ""
    corpus_id = make_vars.get("CORPUS_ID") or ""

    metrics_path = ""
    if dataset_id and view:
        metrics_path = os.path.join("reports", str(dataset_id), str(view), "metrics.json")

    return {
        "dataset_id": dataset_id or "",
        "view": view or "",
        "family": family,
        "model_id": model_id,
        "corpus_id": corpus_id,
        "metrics_path": metrics_path,
    }


def _launch_run(run: RunSpec, log_dir: Path, extra_args: argparse.Namespace) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run.run_id}.log"

    override_items = _build_override_items(run.overrides)
    cmd = [
        sys.executable,
        "-m",
        "scripts.superior.run_single",
        "--exp-id",
        run.exp_id,
        "--run-id",
        run.run_id,
        "--profile",
        run.profile,
        "--stage",
        run.stage,
        "--log-path",
        str(log_path),
    ]
    if extra_args.max_ram_gb:
        cmd.extend(["--max-ram-mb", str(int(float(extra_args.max_ram_gb) * 1024))])

    for k, v in run.make_vars.items():
        cmd.extend(["--make-var", f"{k}={v}"])
    for item in override_items:
        cmd.extend(["--override", item])

    proc = subprocess.Popen(cmd)
    proc.log_path = str(log_path)  # type: ignore[attr-defined]
    return proc


def _current_weight(active: Dict[str, subprocess.Popen], weights: Dict[str, int], plan: Dict[str, RunSpec]) -> int:
    total = 0
    for run_id in active:
        run_spec = plan[run_id]
        weight = weights.get(run_spec.resource_class, DEFAULT_WEIGHTS["light"])
        total += weight
    return total


# ---------------------------------------------------------------------------
# Analysis hooks (minimal V5.4 skeleton)
# ---------------------------------------------------------------------------

def run_analysis_hooks(exp_config: ExpConfig, runs_tsv_path: Path) -> None:
    """Run after-experiment hooks (minimal version)."""
    if not runs_tsv_path.exists():
        return

    runs = read_runs_tsv(runs_tsv_path)
    runs_dir = runs_tsv_path.parent
    hooks = exp_config.analysis_hooks.after_experiment
    if not hooks:
        return

    metrics_rows: List[Dict[str, Any]] = []
    for row in runs.values():
        metrics_path = row.get("metrics_path") or ""
        metrics: Dict[str, Any] = {}
        if metrics_path and os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:
                metrics = {}
        metrics_rows.append({**row, **metrics})

    metrics_global_path = runs_dir / "metrics_global.tsv"
    if metrics_rows:
        _write_tsv(metrics_rows, metrics_global_path, list(metrics_rows[0].keys()))

    for hook in hooks:
        if hook.get("type") == "report_markdown":
            raw_path = hook.get("path") or runs_dir / "report.md"
            report_path = Path(str(raw_path).replace("${exp_id}", exp_config.exp_id))
            _write_report_markdown(Path(report_path), runs)

        elif hook.get("type") == "curves":
            curves_path = runs_dir / "metrics_global.tsv"
            if curves_path.exists():
                # Placeholder: curves generation can be added later (matplotlib)
                pass


def _write_report_markdown(path: Path, runs: Dict[str, Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(runs)
    successes = len([r for r in runs.values() if r.get("status") == "success"])
    failures = total - successes

    lines = [
        f"# Experiment report: {path.stem}",
        "",
        f"Total runs: {total}",
        f"Successes: {successes}",
        f"Failures: {failures}",
        "",
        "## Runs",
    ]
    for row in sorted(runs.values(), key=lambda r: r.get("run_id")):
        lines.append(
            f"- **{row.get('run_id')}** — status: {row.get('status')}, log: {row.get('log_path')}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration entry point
# ---------------------------------------------------------------------------

def orchestrate(args: argparse.Namespace) -> None:
    exp_config = load_exp_config(args.exp_config)

    # CLI overrides scheduler values
    if args.parallel is not None:
        exp_config.scheduler.parallel = int(args.parallel)
    if args.max_ram_gb is not None:
        exp_config.scheduler.max_ram_gb = float(args.max_ram_gb)

    plan = generate_run_plan(exp_config)
    if args.max_runs is not None:
        plan = plan[: int(args.max_runs)]

    exp_dir = Path("superior") / exp_config.exp_id
    logs_dir = exp_dir / "logs"
    runs_tsv = exp_dir / "runs.tsv"
    plan_tsv = exp_dir / "plan.tsv"

    exp_dir.mkdir(parents=True, exist_ok=True)
    write_plan_tsv(plan, plan_tsv)

    if args.dry_run:
        print(f"[DRY-RUN] Generated {len(plan)} runs for exp_id={exp_config.exp_id}")
        for run in plan:
            print(
                f"- {run.run_id} ({run.resource_class}) {run.make_vars} overrides={run.overrides}"
            )
        return

    runs_records = read_runs_tsv(runs_tsv)
    plan_map = {run.run_id: run for run in plan}

    # Optionally skip completed runs
    pending_runs: List[RunSpec] = []
    for run in plan:
        if (
            args.resume
            and run.run_id in runs_records
            and runs_records[run.run_id].get("status") == "success"
        ):
            continue
        pending_runs.append(run)

    active: Dict[str, subprocess.Popen] = {}
    start_times: Dict[str, float] = {}

    while pending_runs or active:
        # Launch new runs if capacity allows
        while pending_runs and len(active) < exp_config.scheduler.parallel:
            current_weight = _current_weight(active, exp_config.scheduler.weights, plan_map)
            next_run = pending_runs[0]
            next_weight = exp_config.scheduler.weights.get(
                next_run.resource_class, DEFAULT_WEIGHTS["light"]
            )
            if current_weight + next_weight > exp_config.scheduler.max_weight:
                break
            pending_runs.pop(0)
            proc = _launch_run(next_run, logs_dir, args)
            active[next_run.run_id] = proc
            start_times[next_run.run_id] = time.time()

            meta = _infer_metadata(next_run.make_vars, next_run.overrides)
            runs_records[next_run.run_id] = {
                "run_id": next_run.run_id,
                "exp_id": next_run.exp_id,
                "profile": next_run.profile,
                "stage": next_run.stage,
                "status": "running",
                "return_code": "",
                "axis_values_json": json.dumps(next_run.axis_values, ensure_ascii=False),
                "make_vars_json": json.dumps(next_run.make_vars, ensure_ascii=False),
                "overrides_json": json.dumps(next_run.overrides, ensure_ascii=False),
                "log_path": getattr(proc, "log_path", ""),
                "started_at": time.time(),
                "finished_at": "",
                "duration_s": "",
                **meta,
            }
            write_runs_tsv(runs_records, runs_tsv)

        # Poll active runs
        time.sleep(0.5)
        finished: List[str] = []
        for run_id, proc in active.items():
            ret = proc.poll()
            if ret is None:
                continue
            finished.append(run_id)
            started_at = start_times.get(run_id, time.time())
            duration = time.time() - started_at
            row = runs_records.get(run_id, {})
            row.update(
                {
                    "status": "success" if ret == 0 else "failed",
                    "return_code": ret,
                    "finished_at": time.time(),
                    "duration_s": duration,
                }
            )
            runs_records[run_id] = row
        for run_id in finished:
            active.pop(run_id, None)
            start_times.pop(run_id, None)
            write_runs_tsv(runs_records, runs_tsv)

    # Analysis hooks after all runs
    run_analysis_hooks(exp_config, runs_tsv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Superior orchestrator (V5)")
    parser.add_argument("--exp-config", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--parallel", type=int, default=None, help="Max parallel runs (default from config)"
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=None,
        help="Max RAM budget (GB) – reserved for future monitoring",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of runs (debug)")
    parser.add_argument("--resume", action="store_true", help="Skip runs already marked success in runs.tsv")
    parser.add_argument("--dry-run", action="store_true", help="Generate plan only, do not execute")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    orchestrate(args)


if __name__ == "__main__":
    main()
