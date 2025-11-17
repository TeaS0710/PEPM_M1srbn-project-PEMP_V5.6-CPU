# scripts/core/core_train.py

import argparse
import csv
import json
import os
import random
import importlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib

from scripts.core.core_utils import (
    resolve_profile_base,
    debug_print_params,
    PIPELINE_VERSION,
)


# ----------------- CLI -----------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="V4 core_train : entraînement multi-familles (spaCy, sklearn, HF, check)"
    )
    ap.add_argument(
        "--profile",
        required=True,
        help="Nom du profil (sans .yml) dans configs/profiles/",
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config (clé=valeur, ex: hardware_preset=lab)",
    )
    ap.add_argument(
        "--only-family",
        choices=["spacy", "sklearn", "hf", "check"],
        help="Limiter l'entraînement à une seule famille (optionnel)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les paramètres résolus",
    )
    return ap.parse_args()


# ----------------- Utils généraux -----------------

def compute_class_weights_from_counts(label_counts: Counter) -> Dict[str, float]:
    """
    Même formule que dans core_prepare : w(label) = n_samples / (n_labels * n_label).

    Utilisé ici pour remplir class_weight des modèles sklearn quand la
    stratégie d'équilibrage est 'class_weights' et que le config demande
    explicitement class_weight: "from_balance".
    """
    total = sum(label_counts.values())
    n_labels = len(label_counts) or 1
    if total <= 0:
        return {lab: 1.0 for lab in label_counts}

    weights: Dict[str, float] = {}
    for lab, c in label_counts.items():
        if c <= 0:
            weights[lab] = 0.0
        else:
            weights[lab] = total / (n_labels * c)
    return weights



def set_blas_threads(n_threads: int) -> None:
    """
    Limiter les threads BLAS (MKL/OPENBLAS/OMP) pour éviter la sur-souscription.
    """
    if n_threads is None or n_threads <= 0:
        return
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n_threads)
    print(f"[core_train] BLAS threads fixés à {n_threads}")


def import_string(path: str):
    """
    Import dynamique d'une classe ou fonction à partir d'une chaîne:
    ex: 'sklearn.svm.LinearSVC' -> class.
    """
    module_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_model_output_dir(params: Dict[str, Any], family: str, model_id: str) -> Path:
    corpus_id = params.get("corpus_id", params["corpus"].get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")
    return Path("models") / corpus_id / view / family / model_id


def load_tsv_dataset(params: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """
    Charger train.tsv et job.tsv depuis data/interim/{corpus_id}/{view}/
    Retourne (train_texts, train_labels, job_texts).
    job_labels ne sont pas strictement nécessaires pour l'entraînement (évent. early stopping),
    on peut les charger plus tard lors de l'évaluation.
    """
    corpus_id = params.get("corpus_id", params["corpus"].get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")
    interim_dir = Path("data") / "interim" / corpus_id / view
    train_path = interim_dir / "train.tsv"
    job_path = interim_dir / "job.tsv"

    if not train_path.exists():
        raise SystemExit(f"[core_train] train.tsv introuvable: {train_path}")

    def read_tsv(path: Path) -> Tuple[List[str], List[str]]:
        texts: List[str] = []
        labels: List[str] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = row.get("text") or ""
                label = row.get("label")
                if not text or not label:
                    continue
                texts.append(text)
                labels.append(label)
        return texts, labels

    train_texts, train_labels = read_tsv(train_path)

    if job_path.exists():
        job_texts, _job_labels = read_tsv(job_path)
    else:
        print(f"[core_train] WARNING: job.tsv introuvable, on utilisera train comme job pour certains usages.")
        job_texts = train_texts

    return train_texts, train_labels, job_texts


def maybe_debug_subsample(
    texts: List[str],
    labels: List[str],
    params: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Si debug_mode=True, limiter la taille du dataset (ex: 1000 docs max).
    """
    if not params.get("debug_mode"):
        return texts, labels

    max_docs = 1000
    if len(texts) <= max_docs:
        return texts, labels

    print(f"[core_train] debug_mode actif : sous-échantillon de {max_docs} docs sur {len(texts)}")
    indices = list(range(len(texts)))
    random.Random(42).shuffle(indices)
    idx_sel = sorted(indices[:max_docs])
    texts_sub = [texts[i] for i in idx_sel]
    labels_sub = [labels[i] for i in idx_sel]
    return texts_sub, labels_sub


def save_meta_model(
    params: Dict[str, Any],
    family: str,
    model_id: str,
    model_dir: Path,
    extra: Dict[str, Any],
) -> None:
    meta = {
        "profile": params.get("profile"),
        "description": params.get("description", ""),
        "corpus_id": params.get("corpus_id", params["corpus"].get("corpus_id")),
        "view": params.get("view"),
        "family": family,
        "model_id": model_id,
        "hardware": params.get("hardware", {}),
        "debug_mode": params.get("debug_mode", False),
        "pipeline_version": PIPELINE_VERSION,
    }
    meta.update(extra)
    meta_path = model_dir / "meta_model.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[core_train] meta_model.json écrit : {meta_path}")


# ----------------- Entraînement spaCy -----------------

def train_spacy_model(params: Dict[str, Any], model_id: str) -> None:
    try:
        import spacy
        from spacy.util import minibatch
        from spacy.tokens import DocBin
    except ImportError:
        raise SystemExit("[core_train] spaCy n'est pas installé, impossible de lancer la famille 'spacy'.")

    models_cfg = params["models_cfg"]["families"]["spacy"][model_id]
    lang = models_cfg.get("lang", "fr")
    epochs = int(models_cfg.get("epochs", 5))
    dropout = float(models_cfg.get("dropout", 0.2))
    arch = models_cfg.get("arch", None)  # "bow" ou "cnn" (info logguée pour l'instant)
    config_template = models_cfg.get("config_template")  # pour une future V4+ avec configs spaCy

    corpus_id = params.get("corpus_id", params.get("corpus", {}).get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")

    # Répertoire où core_prepare a mis les DocBin éventuels
    spacy_proc_dir = Path("data") / "processed" / corpus_id / view / "spacy"

    nlp = spacy.blank(lang)

    train_data: List[Tuple[str, Dict[str, float]]] = []
    labels_set: List[str] = []

    # Chercher tous les DocBin "train*.spacy" (shardés ou non)
    train_docbins = sorted(spacy_proc_dir.glob("train*.spacy"))

    if train_docbins:
        print(f"[core_train:spacy] Utilisation de {len(train_docbins)} DocBin train*.spacy dans {spacy_proc_dir}")
        docs = []
        for path in train_docbins:
            db = DocBin().from_disk(path)
            docs.extend(list(db.get_docs(nlp.vocab)))

        # Sous-échantillon si debug_mode (comme pour le TSV)
        if params.get("debug_mode") and len(docs) > 1000:
            print(f"[core_train:spacy] debug_mode actif : sous-échantillon de 1000 docs sur {len(docs)}")
            docs = docs[:1000]

        labels_set = sorted(
            {lab for doc in docs for lab, val in doc.cats.items() if val}
        )
        for doc in docs:
            # doc.cats est déjà un dict {label: score/bool}
            train_data.append((doc.text, dict(doc.cats)))
        train_source = "docbin"
    else:
        # ---- Fallback TSV (compatible ancien flux V2) ----
        print(f"[core_train:spacy] Aucun DocBin train*.spacy trouvé, fallback sur train.tsv")
        train_texts, train_labels, _job_texts = load_tsv_dataset(params)
        train_texts, train_labels = maybe_debug_subsample(train_texts, train_labels, params)

        labels_set = sorted(set(train_labels))
        for text, label in zip(train_texts, train_labels):
            cats = {lab: (lab == label) for lab in labels_set}
            train_data.append((text, cats))
        train_source = "tsv"

    print(f"[core_train:spacy] Modèle={model_id}, labels={labels_set}, n_train_docs={len(train_data)}")

    # Construction du pipe textcat (simple, multi-classes exclusif pour l'instant)
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat")
    else:
        textcat = nlp.get_pipe("textcat")

    # On part sur un textcat multi-classes exclusif
    for label in labels_set:
        textcat.add_label(label)  # type: ignore[call-arg]

    # Initialisation & entraînement
    optimizer = nlp.begin_training()
    print(f"[core_train:spacy] Entraînement pour {epochs} epochs.")
    for epoch in range(epochs):
        random.shuffle(train_data)
        losses: Dict[str, float] = {}
        batches = minibatch(train_data, size=8)
        for batch in batches:
            texts = [t for t, _ in batch]
            annotations = [{"cats": cats} for _, cats in batch]
            nlp.update(texts, annotations, sgd=optimizer, drop=dropout, losses=losses)
        print(f"[core_train:spacy] Epoch {epoch+1}/{epochs} - pertes={losses}")

    model_dir = get_model_output_dir(params, "spacy", model_id)
    ensure_dir(model_dir)
    nlp.to_disk(model_dir)
    print(f"[core_train:spacy] Modèle spaCy sauvegardé dans {model_dir}")

    save_meta_model(
        params,
        "spacy",
        model_id,
        model_dir,
        extra={
            "epochs": epochs,
            "dropout": dropout,
            "arch": arch,
            "config_template": config_template,
            "labels": labels_set,
            "n_train_docs": len(train_data),
            "train_source": train_source,
        },
    )


# ----------------- Entraînement sklearn -----------------


def train_sklearn_model(params: Dict[str, Any], model_id: str) -> None:
    models_cfg = params["models_cfg"]["families"]["sklearn"][model_id]
    vect_cfg = models_cfg["vectorizer"]
    est_cfg = models_cfg["estimator"]

    vect_class = import_string(vect_cfg["class"])
    est_class = import_string(est_cfg["class"])

    vect_params = dict(vect_cfg.get("params", {}))
    est_params = dict(est_cfg.get("params", {}))

    if params.get("balance_strategy") == "class_weights":
        label_counts = Counter(y_train)
        class_weights = compute_class_weights_from_counts(label_counts)
        if est_params.get("class_weight") == "from_balance":
            est_params = dict(est_params)
            est_params["class_weight"] = class_weights

    # Ajuster n_jobs si possible
    max_procs = params.get("hardware", {}).get("max_procs")
    if max_procs and "n_jobs" in est_params and est_params["n_jobs"] in (None, -1):
        est_params["n_jobs"] = max_procs

    vectorizer = vect_class(**vect_params)
    estimator = est_class(**est_params)

    train_texts, train_labels, job_texts = load_tsv_dataset(params)
    train_texts, train_labels = maybe_debug_subsample(train_texts, train_labels, params)

    print(f"[core_train:sklearn] Modèle={model_id}, {len(train_texts)} docs d'entraînement.")

    X_train = vectorizer.fit_transform(train_texts)
    estimator.fit(X_train, train_labels)

    model_dir = get_model_output_dir(params, "sklearn", model_id)
    ensure_dir(model_dir)
    model_path = model_dir / "model.joblib"
    joblib.dump({"vectorizer": vectorizer, "estimator": estimator}, model_path)
    print(f"[core_train:sklearn] Modèle sklearn sauvegardé dans {model_path}")

    save_meta_model(
        params,
        "sklearn",
        model_id,
        model_dir,
        extra={
            "vectorizer_class": vect_cfg["class"],
            "estimator_class": est_cfg["class"],
            "vectorizer_params": vect_params,
            "estimator_params": est_params,
            "n_train_docs": len(train_texts),
            "n_features": int(getattr(X_train, "shape", (0, 0))[1]),
        },
    )


# ----------------- Entraînement HF (squelette) -----------------


def train_hf_model(params: Dict[str, Any], model_id: str) -> None:
    """
    Entraînement générique HuggingFace (famille 'hf') en mode config-first.

    - Lit les hyperparamètres dans params["models_cfg"]["families"]["hf"][model_id]
    - Lit les données via load_tsv_dataset(params)
    - Ne dépend d'aucune logique spécifique au modèle :
      ajout de modèles via models.yml uniquement.
    """
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import TrainingArguments, Trainer
    except ImportError:
        print("[core_train:hf] Transformers ou torch non installés. Skip HF.")
        return

    models_cfg = params["models_cfg"]["families"]["hf"][model_id]

    model_name = models_cfg.get("model_name")
    if not model_name:
        raise SystemExit(f"[core_train:hf] 'model_name' manquant pour le modèle HF '{model_id}' dans models.yml")

    tokenizer_class_path = models_cfg.get("tokenizer_class", "transformers.AutoTokenizer")
    model_class_path = models_cfg.get("model_class", "transformers.AutoModelForSequenceClassification")
    trainer_params = models_cfg.get("trainer_params", {}) or {}

    # ---------- Données ----------
    train_texts, train_labels_str, _job_texts = load_tsv_dataset(params)
    train_texts, train_labels_str = maybe_debug_subsample(train_texts, train_labels_str, params)

    if not train_texts:
        raise SystemExit("[core_train:hf] Dataset d'entraînement vide.")

    # Mapping label -> id (stable, loggable)
    unique_labels = sorted(set(train_labels_str))
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    train_labels = [label2id[lab] for lab in train_labels_str]

    # ---------- Import dynamique des classes HF ----------
    import importlib

    def import_class(path: str):
        mod_name, cls_name = path.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)

    TokCls = import_class(tokenizer_class_path)
    ModelCls = import_class(model_class_path)

    tokenizer = TokCls.from_pretrained(model_name)
    num_labels = len(unique_labels)
    model = ModelCls.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    max_length = models_cfg.get("max_length") or trainer_params.get("max_length", 256)

    class HFDataset(Dataset):
        def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int):
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            enc = {k: torch.tensor(v) for k, v in enc.items()}
            enc["labels"] = torch.tensor(self.labels[idx])
            return enc

    train_dataset = HFDataset(train_texts, train_labels, tokenizer, max_length)

    # ---------- Répertoires de sortie ----------
    model_dir = get_model_output_dir(params, "hf", model_id)
    ensure_dir(model_dir)
    output_dir = model_dir / "hf_outputs"
    ensure_dir(output_dir)

    # ---------- Hyperparams / hardware (config-first) ----------
    train_batch_size = int(trainer_params.get("per_device_train_batch_size", 8))
    eval_batch_size = int(trainer_params.get("per_device_eval_batch_size", train_batch_size))
    num_train_epochs = float(trainer_params.get("num_train_epochs", 3.0))
    learning_rate = float(trainer_params.get("learning_rate", 2e-5))
    weight_decay = float(trainer_params.get("weight_decay", 0.0))
    warmup_ratio = float(trainer_params.get("warmup_ratio", 0.0))
    grad_accum = int(trainer_params.get("gradient_accumulation_steps", 1))

    from transformers import TrainingArguments, Trainer  # re-import local, safe

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=grad_accum,
        evaluation_strategy="no",   # l'éval se fera dans core_evaluate.py
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print(f"[core_train:hf] Entraînement HF pour '{model_id}' avec {len(train_texts)} docs.")
    trainer.train()

    # Sauvegarde du modèle final + tokenizer dans model_dir
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    extra = {
        "hf_model_name": model_name,
        "label2id": label2id,
        "id2label": id2label,
        "trainer_params": trainer_params,
        "n_train_docs": len(train_texts),
    }
    save_meta_model(params, "hf", model_id, model_dir, extra=extra)



# ----------------- Entraînement "check" -----------------


def train_check_model(params: Dict[str, Any], model_id: str = "check_default") -> None:
    """
    Famille 'check' vue comme un "pseudo-modèle" :
    il peut générer des stats, des sanity checks, etc., et écrire un meta_model.json.
    Pour l'instant on se contente de consigner les stats de base.
    """
    train_texts, train_labels, job_texts = load_tsv_dataset(params)
    labels_set = sorted(set(train_labels))
    label_counts = {l: train_labels.count(l) for l in labels_set}

    model_dir = get_model_output_dir(params, "check", model_id)
    ensure_dir(model_dir)

    save_meta_model(
        params,
        "check",
        model_id,
        model_dir,
        extra={
            "n_train_docs": len(train_texts),
            "n_labels": len(labels_set),
            "label_counts": label_counts,
            "note": "Famille 'check' = modèle virtuel pour sanity checks / stats",
        },
    )
    print(f"[core_train:check] Checks de base consignés dans {model_dir}")


# ----------------- main -----------------


def main() -> None:
    args = parse_args()
    params = resolve_profile_base(args.profile, args.override)

    if args.verbose:
        debug_print_params(params)

    # Seed de base pour une reproductibilité minimaliste
    random.seed(42)

    hw = params.get("hardware", {})
    blas_threads = hw.get("blas_threads", 1)
    set_blas_threads(blas_threads)

    families = params.get("families", []) or []
    if args.only_family and args.only_family in families:
        families = [args.only_family]

    # Construire la liste des modèles à entraîner
    models_to_train: List[Dict[str, Any]] = []

    if "check" in families:
        # Pour l'instant un seul pseudo-modèle check_default
        models_to_train.append({"family": "check", "model_id": "check_default"})

    if "spacy" in families:
        for mid in params.get("models_spacy", []) or []:
            models_to_train.append({"family": "spacy", "model_id": mid})

    if "sklearn" in families:
        for mid in params.get("models_sklearn", []) or []:
            models_to_train.append({"family": "sklearn", "model_id": mid})

    if "hf" in families:
        for mid in params.get("models_hf", []) or []:
            models_to_train.append({"family": "hf", "model_id": mid})

    if not models_to_train:
        print(f"[core_train] Aucun modèle à entraîner pour le profil '{params.get('profile')}'. Rien à faire.")
        return

    print("[core_train] Modèles à entraîner :")
    for m in models_to_train:
        print(f"  - {m['family']}::{m['model_id']}")

    # Entraînement
    for m in models_to_train:
        family = m["family"]
        mid = m["model_id"]
        if family == "spacy":
            train_spacy_model(params, mid)
        elif family == "sklearn":
            train_sklearn_model(params, mid)
        elif family == "hf":
            train_hf_model(params, mid)
        elif family == "check":
            train_check_model(params, mid)
        else:
            print(f"[core_train] WARNING: famille inconnue '{family}', ignorée.")


if __name__ == "__main__":
    main()
