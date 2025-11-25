from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from scripts.core.core_utils import load_yaml

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


# Data classes (see dev_V5.md)

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
    approx_ram_per_class: Dict[str, float] = field(default_factory=dict)


@dataclass
class OomPolicy:
    on_oom: str = "skip"        # "skip" | "backoff" | "stop
    backoff_factor: float = 0.5


@dataclass
class EarlyStopConfig:
    enabled: bool = False
    min_accuracy: float = 0.0
    min_macro_f1: float = 0.0
    apply_to_families: List[str] = field(default_factory=list)


@dataclass
class SafetyConfig:
    enable_hard_ram_limit: bool = False
    hard_limit_mb: Optional[int] = None

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
    oom_policy: OomPolicy
    early_stop: EarlyStopConfig
    safety: SafetyConfig



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



# Helpers
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
    """
    Sérialise une valeur d'override pour la ligne de commande.

    - Pour les dict/list, on utilise un JSON compact (sans espaces après les virgules)
      afin d'éviter que `make` / le shell ne découpent l'override en plusieurs tokens.
    """
    if isinstance(value, (dict, list)):
        # JSON compact: '["web1","asr1"]'
        # → après passage shell: [web1,asr1] (sans guillemets, mais une seule "word")
        return json.dumps(value, separators=(",", ":"))
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


# Loading / parsing experiment configuration

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
        approx_ram_per_class=scheduler_cfg.get("approx_ram_per_class") or {},
    )

    oom_cfg = raw.get("oom_policy") or {}
    oom_policy = OomPolicy(
        on_oom=oom_cfg.get("on_oom", "skip"),
        backoff_factor=float(oom_cfg.get("backoff_factor", 0.5)),
    )

    es_cfg = raw.get("early_stop") or {}
    early_stop = EarlyStopConfig(
        enabled=es_cfg.get("enabled", False),
        min_accuracy=float(es_cfg.get("min_accuracy", 0.0)),
        min_macro_f1=float(es_cfg.get("min_macro_f1", 0.0)),
        apply_to_families=es_cfg.get("apply_to_families") or [],
    )

    safety_cfg = raw.get("safety") or {}
    safety = SafetyConfig(
        enable_hard_ram_limit=safety_cfg.get("enable_hard_ram_limit", False),
        hard_limit_mb=safety_cfg.get("hard_limit_mb"),
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
        oom_policy=oom_policy,
        early_stop=early_stop,
        safety=safety,
    )



# Plan generation


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



# Plan / run persistence (TSV helpers)

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
    "train_prop",
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



# Scheduler helpers


def _build_override_items(overrides: Dict[str, Any]) -> List[str]:
    flat: Dict[str, Any] = {}
    for key, value in overrides.items():
        flat.update(_flatten_overrides(key, value))
    return [f"{k}={_format_override_value(v)}" for k, v in flat.items()]


def _lookup_nested(overrides: Dict[str, Any], dotted_key: str) -> Optional[Any]:
    """Best-effort nested lookup using dotted paths."""

    if dotted_key in overrides:
        return overrides.get(dotted_key)
    parts = dotted_key.split(".")
    current: Any = overrides
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _first_in_list(val: Any) -> Optional[str]:
    if isinstance(val, list) and val:
        return str(val[0])
    if isinstance(val, str):
        if "," in val:
            return val.split(",", 1)[0]
        return val
    return None


def _infer_metadata(
    make_vars: Dict[str, str], overrides: Dict[str, Any], profile: Optional[str] = None
) -> Dict[str, Any]:
    dataset_id = (
        overrides.get("dataset_id")
        or make_vars.get("DATASET_ID")
        or make_vars.get("CORPUS_ID")
        or ""
    )

    view = overrides.get("view") or _lookup_nested(overrides, "ideology.view") or ""

    family = make_vars.get("FAMILY") or ""

    model_id: Optional[str] = None
    if family:
        model_id = make_vars.get("MODEL_ID") or make_vars.get("model_id")
        if not model_id:
            if family == "sklearn":
                model_id = _first_in_list(overrides.get("models_sklearn"))
            elif family == "spacy":
                model_id = _first_in_list(overrides.get("models_spacy"))
            elif family == "hf":
                model_id = _first_in_list(overrides.get("models_hf"))

    corpus_id = make_vars.get("CORPUS_ID") or ""
    train_prop = make_vars.get("TRAIN_PROP") or ""

    metrics_path = ""
    if dataset_id and view and family and model_id:
        metrics_path = os.path.join(
            "reports",
            str(dataset_id),
            str(view),
            str(family),
            str(model_id),
            "metrics.json",
        )

    return {
        "dataset_id": dataset_id or "",
        "view": view or "",
        "family": family or "",
        "model_id": model_id or "",
        "corpus_id": corpus_id or "",
        "train_prop": train_prop or "",
        "metrics_path": metrics_path,
    }


def _launch_run(run: RunSpec, log_dir: Path, safety: SafetyConfig) -> subprocess.Popen:
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
    # Hard RAM limit optionnelle, contrôlée par la section safety de la config d'expérience
    if safety.enable_hard_ram_limit and safety.hard_limit_mb is not None:
        cmd.extend(["--max-ram-mb", str(int(safety.hard_limit_mb))])

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

def _current_soft_ram_gb(
    active: Dict[str, subprocess.Popen],
    scheduler: SchedulerConfig,
    plan: Dict[str, RunSpec],
) -> float:
    total = 0.0
    if not scheduler.approx_ram_per_class:
        return total
    for run_id in active:
        spec = plan[run_id]
        rc = spec.resource_class or "light"
        total += float(scheduler.approx_ram_per_class.get(rc, 0.0))
    return total



# Analysis hooks (minimal V5.4 skeleton)


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
                print(f"[analysis_hooks] Failed to load metrics at {metrics_path}, skipping.")
                metrics = {}
        else:
            if metrics_path:
                print(
                    f"[analysis_hooks] metrics.json not found at {metrics_path} for run {row.get('run_id')}"
                )
        enriched_row = {**row, **metrics}
        # Propagate TRAIN_PROP if available in make_vars_json to ease plotting hooks
        make_vars_json = row.get("make_vars_json")
        train_prop_val: str | float = ""
        if make_vars_json:
            try:
                make_vars_data = json.loads(make_vars_json)
                if isinstance(make_vars_data, dict) and "TRAIN_PROP" in make_vars_data:
                    train_prop_val = make_vars_data.get("TRAIN_PROP")
            except Exception:
                pass
        enriched_row["TRAIN_PROP"] = train_prop_val
        metrics_rows.append(enriched_row)

    metrics_global_path = runs_dir / "metrics_global.tsv"
    if metrics_rows:
        base_columns = list(RUN_COLUMNS)
        extra_columns = set()
        for row in metrics_rows:
            extra_columns.update(set(row.keys()) - set(base_columns))
        columns = base_columns + sorted(extra_columns)
        _write_tsv(metrics_rows, metrics_global_path, columns)

    for hook in hooks:
        if hook.get("type") == "report_markdown":
            raw_path = hook.get("path") or runs_dir / "report.md"
            report_path = Path(str(raw_path).replace("${exp_id}", exp_config.exp_id))
            _write_report_markdown(Path(report_path), runs)

        elif hook.get("type") == "curves":
            curves_path = runs_dir / "metrics_global.tsv"
            if not curves_path.exists():
                print(f"[analysis_hooks] metrics_global.tsv missing at {curves_path}, skipping curves")
                continue

            if plt is None:
                print("[superior] matplotlib non disponible, hook 'curves' ignoré")
                continue

            with curves_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                curves_rows_raw = list(reader)

            if not curves_rows_raw:
                print("[analysis_hooks] metrics_global.tsv is empty, skipping curves")
                continue

            curves_rows: List[Dict[str, Any]] = []
            for row in curves_rows_raw:
                parsed_row: Dict[str, Any] = {}
                for key, value in row.items():
                    if value in (None, ""):
                        parsed_row[key] = ""
                        continue
                    try:
                        parsed_row[key] = float(value)
                    except ValueError:
                        parsed_row[key] = value
                curves_rows.append(parsed_row)

            metrics = hook.get("metrics") or []
            x_axis = hook.get("x_axis")
            group_by = hook.get("group_by") or []

            all_columns = set().union(*[row.keys() for row in curves_rows])
            required_columns = set(metrics + ([x_axis] if x_axis else []) + group_by)
            missing = [col for col in required_columns if col and col not in all_columns]
            if missing:
                print(
                    f"[analysis_hooks] Missing columns for curves ({', '.join(missing)}), skipping hook"
                )
                continue

            plots_dir = runs_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            for metric in metrics:
                grouped: Dict[tuple, List[tuple]] = {}
                for row in curves_rows:
                    x_val = row.get(x_axis) if x_axis else None
                    y_val = row.get(metric)
                    if x_val in (None, "") or y_val in (None, ""):
                        continue
                    if not isinstance(x_val, (int, float)) or not isinstance(y_val, (int, float)):
                        continue
                    group_key = tuple(row.get(col, "") for col in group_by)
                    grouped.setdefault(group_key, []).append((x_val, y_val))

                if not grouped:
                    print(
                        f"[analysis_hooks] No data for metric '{metric}' with x_axis '{x_axis}', skipping"
                    )
                    continue

                plt.figure()
                for group_key, points in grouped.items():
                    points_sorted = sorted(points, key=lambda p: p[0])
                    xs, ys = zip(*points_sorted)
                    label = ", ".join(str(v) for v in group_key if v)
                    plt.plot(xs, ys, marker="o", label=label or None)

                plt.xlabel(x_axis)
                plt.ylabel(metric)
                plt.title(f"{metric} vs {x_axis}")
                if group_by:
                    plt.legend()
                plt.grid(True, linestyle=":", alpha=0.4)
                plt.tight_layout()

                plot_path = runs_dir / "plots" / f"{metric}__vs__{x_axis}.png"
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_path)
                plt.close()


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



def _make_oom_key(run: RunSpec) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    return (run.stage, tuple(sorted(run.axis_values.items())))


def _apply_oom_policy(
    run: RunSpec,
    status: str,
    oom_policy: OomPolicy,
    pending_runs: List[RunSpec],
    runs_records: Dict[str, Dict[str, Any]],
) -> None:
    if status != "oom":
        return
    if not oom_policy:
        return

    key = _make_oom_key(run)

    if oom_policy.on_oom not in {"skip", "backoff", "stop"}:
        return

    if oom_policy.on_oom in {"skip", "backoff"}:
        # On marque comme skip les futures runs avec la même config d'axes
        to_skip: List[RunSpec] = []
        for spec in pending_runs:
            if _make_oom_key(spec) == key:
                to_skip.append(spec)
        for spec in to_skip:
            pending_runs.remove(spec)
            row = runs_records.get(spec.run_id, {})
            row.update(
                {
                    "run_id": spec.run_id,
                    "exp_id": spec.exp_id,
                    "profile": spec.profile,
                    "stage": spec.stage,
                    "status": "skipped_oom_backoff"
                    if oom_policy.on_oom == "backoff"
                    else "skipped_oom",
                    "return_code": "",
                }
            )
            runs_records[spec.run_id] = row

    elif oom_policy.on_oom == "stop":
        # On abandonne le reste du plan
        for spec in pending_runs:
            row = runs_records.get(spec.run_id, {})
            row.update(
                {
                    "run_id": spec.run_id,
                    "exp_id": spec.exp_id,
                    "profile": spec.profile,
                    "stage": spec.stage,
                    "status": "aborted_oom_policy",
                    "return_code": "",
                }
            )
            runs_records[spec.run_id] = row
        pending_runs.clear()


def _apply_early_stop(
    run: RunSpec,
    status: str,
    early_stop: EarlyStopConfig,
    runs_records: Dict[str, Dict[str, Any]],
    exp_dir: Path,
) -> None:
    if not early_stop.enabled:
        return
    if status != "success":
        return

    row = runs_records.get(run.run_id)
    if not row:
        return

    family = row.get("family")
    if family not in (early_stop.apply_to_families or []):
        return

    metrics_path = row.get("metrics_path")
    if not metrics_path or not os.path.exists(metrics_path):
        return

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        return

    acc = float(metrics.get("accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))

    quality_ok = True
    flags: List[str] = []
    if acc < early_stop.min_accuracy:
        quality_ok = False
        flags.append("low_accuracy")
    if macro_f1 < early_stop.min_macro_f1:
        quality_ok = False
        flags.append("low_macro_f1")

    row["quality_ok"] = str(quality_ok)
    row["quality_flags"] = ",".join(flags)
    runs_records[run.run_id] = row

    if not quality_ok:
        log_path = exp_dir / "early_stop.log"
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(
                f"{run.run_id}\tfamily={family}\taccuracy={acc:.4f}\t"
                f"macro_f1={macro_f1:.4f}\tflags={','.join(flags)}\n"
            )




# Orchestration entry point


def orchestrate(
    exp_config_path: str,
    parallel: Optional[int] = None,
    max_ram_gb: Optional[float] = None,
    max_runs: Optional[int] = None,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    exp_config = load_exp_config(exp_config_path)

    # CLI overrides scheduler values
    if parallel is not None:
        exp_config.scheduler.parallel = int(parallel)
    if max_ram_gb is not None:
        exp_config.scheduler.max_ram_gb = float(max_ram_gb)

    plan = generate_run_plan(exp_config)
    if max_runs is not None:
        plan = plan[: int(max_runs)]

    exp_dir = Path("superior") / exp_config.exp_id
    logs_dir = exp_dir / "logs"
    runs_tsv = exp_dir / "runs.tsv"
    plan_tsv = exp_dir / "plan.tsv"

    exp_dir.mkdir(parents=True, exist_ok=True)
    write_plan_tsv(plan, plan_tsv)

    if dry_run:
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
            resume
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

            # Limite RAM "soft" basée sur approx_ram_per_class et max_ram_gb
            if exp_config.scheduler.max_ram_gb is not None and exp_config.scheduler.approx_ram_per_class:
                current_ram = _current_soft_ram_gb(active, exp_config.scheduler, plan_map)
                rc = next_run.resource_class or "light"
                needed = float(exp_config.scheduler.approx_ram_per_class.get(rc, 0.0))
                if current_ram + needed > exp_config.scheduler.max_ram_gb:
                    # On attend qu'un run se termine avant de lancer le suivant
                    break

            pending_runs.pop(0)

            # LOG VERBEUX LANCEMENT D'UN RUN
            family = next_run.make_vars.get("FAMILY", "")
            model = next_run.make_vars.get("MODEL_ID", "")
            print(
                f"[superior] Launching {next_run.run_id} "
                f"(stage={next_run.stage}, profile={next_run.profile}, "
                f"axes={next_run.axis_values}, FAMILY={family}, MODEL={model})"
            )


            proc = _launch_run(next_run, logs_dir, exp_config.safety)
            active[next_run.run_id] = proc
            start_times[next_run.run_id] = time.time()

            meta = _infer_metadata(next_run.make_vars, next_run.overrides, next_run.profile)
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
            status = "success"
            if ret == 99:
                status = "oom"
            elif ret != 0:
                status = "failed"
            row.update(
                {
                    "status": status,
                    "return_code": ret,
                    "finished_at": time.time(),
                    "duration_s": duration,
                }
            )
            runs_records[run_id] = row
            # Politique OOM + early-stop (log-only)
            run_spec = plan_map[run_id]
            _apply_oom_policy(run_spec, status, exp_config.oom_policy, pending_runs, runs_records)
            _apply_early_stop(run_spec, status, exp_config.early_stop, runs_records, exp_dir)

        for run_id in finished:
            active.pop(run_id, None)
            start_times.pop(run_id, None)
            write_runs_tsv(runs_records, runs_tsv)

    # Analysis hooks after all runs
    run_analysis_hooks(exp_config, runs_tsv)



# CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Superior orchestrator (V5)")
    parser.add_argument("--exp-config", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Max parallel runs (override scheduler.parallel from config)",
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=None,
        help=(
            "Soft RAM budget (GB) utilisé par le scheduler pour limiter le "
            "parallélisme (pas une limite de kill; override scheduler.max_ram_gb)."
        ),
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of runs (debug)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs déjà marqués success dans runs.tsv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate plan only, do not execute",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    orchestrate(
        exp_config_path=args.exp_config,
        parallel=args.parallel,
        max_ram_gb=args.max_ram_gb,
        max_runs=args.max_runs,
        resume=args.resume,
        dry_run=args.dry_run,
    )



if __name__ == "__main__":
    main()
