from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import spacy

from scripts.core.core_evaluate import (
    compute_basic_metrics,
    compute_grouped_metrics,
    eval_spacy_model,
    group_indices_by_field,
)
from scripts.core.core_prepare import write_tsv
from scripts.core.core_utils import resolve_profile_base


def test_resolve_single_profile_defaults():
    params = resolve_profile_base("ideo_quick")
    assert params["merge_mode"] == "single"
    assert params["dataset_id"] == params["corpus_id"]
    assert params["corpora"] and len(params["corpora"]) == 1


def test_resolve_multi_profile():
    params = resolve_profile_base("multi_demo")
    assert len(params["corpora"]) == 2
    assert params["dataset_id"] == "web1_web2"
    assert params["source_field"] == "corpus_id"
    assert params["corpus_id"] == "web1_web2"


def test_write_tsv_includes_source_column(tmp_path: Path):
    docs = [
        {
            "id": "doc1",
            "label": "a",
            "label_raw": "a",
            "text": "hello",
            "modality": "web",
            "meta": {"corpus_id": "web1"},
        },
        {
            "id": "doc2",
            "label": "b",
            "label_raw": "b",
            "text": "world",
            "modality": "web",
            "meta": {"corpus_id": "web2"},
        },
    ]
    tsv_path = tmp_path / "sample.tsv"
    write_tsv(str(tsv_path), docs, source_field="corpus_id")

    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    assert reader.fieldnames and "corpus_id" in reader.fieldnames
    assert rows[0]["corpus_id"] == "web1"
    assert rows[1]["corpus_id"] == "web2"


def test_group_metrics_by_field():
    rows = [
        {"corpus_id": "web1"},
        {"corpus_id": "web1"},
        {"corpus_id": "web2"},
    ]
    y_true = ["a", "b", "a"]
    y_pred = ["a", "b", "b"]

    groups = group_indices_by_field(rows, "corpus_id")
    assert set(groups.keys()) == {"web1", "web2"}

    metrics_by_fields = compute_grouped_metrics(rows, y_true, y_pred, ["corpus_id"])
    metrics_by = metrics_by_fields.get("corpus_id", {})

    assert set(metrics_by.keys()) == {"web1", "web2"}
    assert metrics_by["web1"]["accuracy"] == 1.0
    assert metrics_by["web2"]["accuracy"] == 0.0


def test_spacy_compare_by_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_id = "web1_web2_test"
    view = "demo"
    model_id = "dummy"

    # Préparer répertoires
    interim_dir = tmp_path / "data" / "interim" / dataset_id / view
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Écrire job.tsv avec champ corpus_id
    job_tsv = interim_dir / "job.tsv"
    with job_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["text", "label", "corpus_id"])
        writer.writeheader()
        writer.writerow({"text": "hello web1", "label": "a", "corpus_id": "web1"})
        writer.writerow({"text": "hello web2", "label": "b", "corpus_id": "web2"})

    # Créer un petit modèle spaCy qui prédit toujours "a"
    nlp = spacy.blank("en")

    @nlp.component("predict_a")
    def predict_a(doc):
        doc.cats["a"] = 1.0
        doc.cats["b"] = 0.0
        return doc

    nlp.add_pipe("predict_a")

    model_dir = tmp_path / "models" / dataset_id / view / "spacy" / model_id / "model-best"
    model_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(model_dir)

    params = {
        "dataset_id": dataset_id,
        "corpus_id": "web1",  # fallback
        "view": view,
        "analysis": {"compare_by": ["corpus_id"]},
    }

    monkeypatch.chdir(tmp_path)

    eval_spacy_model(params, model_id)

    reports_dir = tmp_path / "reports" / dataset_id / view / "spacy" / model_id
    metrics_by_path = reports_dir / "metrics_by_corpus_id.json"
    assert metrics_by_path.exists()

    with metrics_by_path.open("r", encoding="utf-8") as f:
        metrics_by = json.load(f)

    assert set(metrics_by.keys()) == {"web1", "web2"}
    assert "accuracy" in metrics_by["web1"]
    assert "macro_f1" in metrics_by["web2"]


if __name__ == "__main__":
    pytest.main([__file__])
