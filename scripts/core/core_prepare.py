# scripts/core/core_prepare.py

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import xml.etree.ElementTree as ET

import spacy

from scripts.core.core_utils import (
    resolve_profile_base,
    load_label_map,
    debug_print_params,
    PIPELINE_VERSION,
)


# ----------------- CLI -----------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="V4 core_prepare : TEI -> TSV -> formats (multi-famille)"
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
        help="Override config (clé=valeur, ex: train_prop=0.7, corpus_id=web2)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Ne produit pas les fichiers, imprime seulement les stats de la vue",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les paramètres résolus",
    )
    return ap.parse_args()


# ----------------- Tokenizers & poids de classe -----------------


def _tokenize_whitespace(text: str) -> List[str]:
    """Tokenisation naïve sur espaces (comportement historique)."""
    return text.split()


def _tokenize_simple(text: str) -> List[str]:
    """
    Tokenisation simple : sépare les mots alphanumériques et la ponctuation.
    Utile pour comparer l'effet d'un tokeniseur plus fin.
    """
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def count_tokens(text: str, tokenizer_name: str = "whitespace") -> int:
    """Compter les tokens selon une stratégie configurée."""
    if not text:
        return 0
    if tokenizer_name == "simple":
        return len(_tokenize_simple(text))
    # Valeur par défaut : whitespace
    return len(_tokenize_whitespace(text))


_SPACY_XX_NLP = None


def get_spacy_xx():
    """Retourner un nlp spaCy multilingue 'xx', initialisé une seule fois."""
    global _SPACY_XX_NLP
    if _SPACY_XX_NLP is not None:
        return _SPACY_XX_NLP
    if spacy is None:
        print("[core_prepare] WARNING: spaCy n'est pas disponible, tokenizer 'spacy_xx' indisponible.")
        return None
    try:
        _SPACY_XX_NLP = spacy.blank("xx")
        return _SPACY_XX_NLP
    except Exception as e:
        print(f"[core_prepare] WARNING: impossible de charger spaCy 'xx' : {e}")
        _SPACY_XX_NLP = None
        return None

def get_default_spacy_lang(params: Dict[str, Any]) -> str:
    """Choisir une langue par défaut pour la construction des DocBin spaCy.

    On essaie d'abord de regarder dans models.yml (famille spacy),
    via les modèles référencés par le profil. Sinon, on tombe sur 'fr'.
    """
    families_cfg = params.get("models_cfg", {}).get("families", {})
    spacy_models = families_cfg.get("spacy", {})
    for mid in (params.get("models_spacy") or []):
        cfg = spacy_models.get(mid)
        if cfg and cfg.get("lang"):
            return cfg["lang"]
    return "fr"

def count_tokens(text: str, tokenizer_name: str = "split") -> int:
    """Compter les tokens selon la stratégie configurée.

    tokenizer_name :
      - "split" / None : split() whitespace (comportement historique)
      - "simple"       : tokeniseur regex (sépare la ponctuation)
      - "spacy_xx"     : tokenizer spaCy blank("xx")
      - "spacy"        : alias de "spacy_xx"
    """
    if not text:
        return 0

    name = (tokenizer_name or "split").lower()

    # Tokeniseur spaCy
    if name in ("spacy_xx", "spacy"):
        nlp = get_spacy_xx()
        if nlp is not None:
            # Utiliser le tokenizer spaCy, sans pipeline complète
            doc = nlp.make_doc(text)
            return len(doc)
        # fallback si spaCy indisponible : split whitespace
        return len(_tokenize_whitespace(text))

    # Tokeniseur "simple" regex
    if name == "simple":
        return len(_tokenize_simple(text))

    # Par défaut : split() whitespace
    return len(_tokenize_whitespace(text))



def compute_class_weights_from_counts(label_counts: Counter) -> Dict[str, float]:
    """
    Calculer des poids de classe de type 'balanced' (sklearn) :

        w(label) = n_samples / (n_labels * n_label)

    Ne modifie pas les docs, mais donne des poids utilisables
    par certains modèles (sklearn, HF, etc.).
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



# ----------------- TEI helpers -----------------

def _token_count(s: str) -> int:
    """Compter grossièrement les tokens comme dans V2 (split() whitespace)."""
    return len(s.split()) if s else 0


def _local_name(tag: str) -> str:
    """Extraire le nom local d'un tag XML (sans namespace)."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def iter_tei_docs(tei_path: str) -> Iterable[ET.Element]:
    """
    Itérer sur les documents TEI.

    NOTE : Ici on considère que chaque <text> correspond à un document.
    Adapte cette fonction si ton schéma TEI est différent
    (ex: <TEI> par doc, <div type='article'>, etc.).
    """
    context = ET.iterparse(tei_path, events=("end",))
    for event, elem in context:
        if _local_name(elem.tag) == "text":
            yield elem
            # Libérer la mémoire
            elem.clear()


def extract_term(elem: ET.Element, term_type: str) -> Optional[str]:
    """
    Extraire le texte d'un <term type="xxx"> dans l'élément ou ses descendants.

    Utilisé pour les labels (ideology, crawl, ...) et pour modality.
    """
    for t in elem.iter():
        if _local_name(t.tag) == "term" and t.get("type") == term_type:
            if t.text:
                txt = t.text.strip()
                if txt:
                    return txt
    return None


def extract_doc_id(elem: ET.Element, fallback_idx: int) -> str:
    """
    Essayer de récupérer un identifiant de document dans les attributs TEI,
    sinon utiliser un index numérique.
    """
    # Exemples possibles: @xml:id sur <text>, @n, etc.
    xml_id = elem.get("{http://www.w3.org/XML/1998/namespace}id") or elem.get("xml:id")
    if xml_id:
        return str(xml_id)
    n_attr = elem.get("n")
    if n_attr:
        return str(n_attr)
    return f"doc_{fallback_idx}"


def extract_text(elem: ET.Element) -> str:
    """
    Extraire le texte brut du document TEI.
    Ici c'est très basique : concatenation de tous les .itertext().
    À adapter si besoin (ex: ignorer certaines zones).
    """
    text = " ".join(t.strip() for t in elem.itertext() if t.strip())
    return text


def extract_modality(elem: ET.Element, params: Dict[str, Any]) -> str:
    """
    Extraire la modalité du document :
      1) d'abord à partir de <term type="modality">...,
      2) sinon via default_modality du corpus,
      3) sinon 'unknown'.
    """
    term_mod = extract_term(elem, "modality")
    if term_mod:
        return term_mod

    corpus_default = params.get("corpus", {}).get("default_modality")
    if corpus_default:
        return corpus_default

    return "unknown"


def extract_label_raw(elem: ET.Element, label_field: Optional[str]) -> Optional[str]:
    """
    Extraire le label brut à partir du TEI, en fonction de label_field.
    Par défaut, on cherche un <term type='label_field'>.

    Ex:
      label_field = "ideology" -> <term type="ideology">gauche</term>
      label_field = "crawl"    -> <term type="crawl">X</term>
    """
    if not label_field:
        return None
    return extract_term(elem, label_field)


# ----------------- Vue : TEI -> TSV -----------------


def build_view(params: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """
    Construire la vue supervisée (train.tsv, job.tsv, meta_view.json).
    Retourne un dict de meta pour usage ultérieur (log ou tests).
    """
    corpus = params["corpus"]
    tei_path = corpus["corpus_path"]
    label_field = params.get("label_field")
    label_map_path = params.get("label_map")
    label_map = None
    if label_map_path:
        label_map = load_label_map(label_map_path)

    modality_filter = params.get("modality")  # ex: "web", "asr", etc.
    train_prop = float(params.get("train_prop", 0.8))
    min_chars = params.get("min_chars", 0) or 0
    max_tokens = params.get("max_tokens")  # peut rester None
    tokenizer_name = params.get("tokenizer", "split")

    balance_strategy = params.get("balance_strategy", "none")
    balance_preset = params.get("balance_preset")

    corpus_id = params.get("corpus_id", corpus.get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")

    interim_dir = os.path.join("data", "interim", corpus_id, view)
    os.makedirs(interim_dir, exist_ok=True)

    # Collecte des docs (version V1 : en RAM ; on optimisera plus tard en streaming + sharding)
    docs: List[Dict[str, Any]] = []
    # Construire la distribution des labels
    label_counts = Counter(doc["label"] for doc in docs)
    label_counts_before = Counter(label_counts)

    if not docs:
        raise RuntimeError("Aucun document valide après filtrage.")
    modality_counts = Counter()

    print(f"[core_prepare] Lecture TEI: {tei_path}")
    idx = 0
    for elem in iter_tei_docs(tei_path):
        idx += 1
        doc_id = extract_doc_id(elem, idx)
        text = extract_text(elem)

        if len(text) < min_chars:
            continue

        # compter les tokens selon le tokenizer configuré
        tokens = count_tokens(text, tokenizer_name)

        # Filtre max_tokens basé sur ce tokenizer
        if max_tokens and tokens > max_tokens:
            continue


        doc_modality = extract_modality(elem, params)
        modality_counts[doc_modality] += 1

        if modality_filter and doc_modality != modality_filter:
            # On ne retient pas les docs d'autres modalités
            continue

        label_raw = extract_label_raw(elem, label_field)
        if not label_raw:
            # Document sans label : pour l'instant on le jette
            continue

        if label_map:
            mapped = label_map.get(label_raw)
            if not mapped:
                # label brut inconnu dans le mapping -> on jette le doc
                continue
            label = mapped
        else:
            label = label_raw

        label_counts[label] += 1

        docs.append(
            {
                "id": doc_id,
                "label": label,
                "label_raw": label_raw,
                "text": text,
                "modality": doc_modality,
                "tokens": tokens,  # pour cap_tokens, stats, etc.
            }
        )

    print(f"[core_prepare] Docs retenus avant équilibrage: {len(docs)}")
    print(f"[core_prepare] Répartition labels (avant balance): {label_counts}")
    print(f"[core_prepare] Répartition modalités (tous docs vus): {modality_counts}")

    if not docs:
        raise SystemExit("[core_prepare] Aucun document valide après filtrage. Vérifie ton profil/config.")

    # Appliquer l'équilibrage
    docs_balanced, label_counts_balanced = apply_balance(
        docs, params, label_counts
    )


    print(f"[core_prepare] Docs après équilibrage ({balance_strategy}): {len(docs_balanced)}")
    print(f"[core_prepare] Répartition labels (après balance): {label_counts_balanced}")

    # Split train/job
    seed = int(params.get("seed", 42))
    random.Random(seed).shuffle(docs_balanced)
    n_total = len(docs_balanced)
    n_train = int(round(train_prop * n_total))
    train_docs = docs_balanced[:n_train]
    job_docs = docs_balanced[n_train:]

    print(f"[core_prepare] Split train/job avec train_prop={train_prop}")
    print(f"  -> train: {len(train_docs)} docs")
    print(f"  -> job  : {len(job_docs)} docs")

    meta = {
        "profile": params.get("profile"),
        "corpus_id": corpus_id,
        "view": view,
        "modality_filter": modality_filter,
        "label_field": label_field,
        "label_map": label_map_path,
        "balance_strategy": balance_strategy,
        "balance_preset": balance_preset,
        "train_prop": train_prop,
        "seed": seed,
        "tokenizer": tokenizer_name,
        "n_docs_raw": len(docs),
        "n_docs_train": len(train_docs),
        "n_docs_job": len(job_docs),
        "label_counts_before": dict(label_counts),
        "label_counts_after": dict(label_counts_balanced),
        "modality_counts": dict(modality_counts),
        "params_hardware": params.get("hardware", {}),
        "pipeline_version": PIPELINE_VERSION,
        "n_docs_balanced": len(docs_balanced),
        "label_counts_before": dict(label_counts_before),
        "label_counts_after": dict(label_counts_balanced),

    }

    # Si on utilise une stratégie par poids, on loggue aussi les poids par label
    class_weights = params.get("class_weights")
    if class_weights is not None:
        meta["label_weights"] = class_weights


    if dry_run:
        print("[core_prepare] Dry-run activé, aucun TSV écrit.")
        return meta

    # Écriture TSV
    train_path = os.path.join(interim_dir, "train.tsv")
    job_path = os.path.join(interim_dir, "job.tsv")
    meta_path = os.path.join(interim_dir, "meta_view.json")

    write_tsv(train_path, train_docs)
    write_tsv(job_path, job_docs)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[core_prepare] train.tsv écrit : {train_path}")
    print(f"[core_prepare] job.tsv écrit   : {job_path}")
    print(f"[core_prepare] meta_view.json  : {meta_path}")

    return meta


def write_tsv(path: str, docs: List[Dict[str, Any]]) -> None:
    """Écrire une liste de docs en TSV (id, label, label_raw, modality, text)."""
    fieldnames = ["id", "label", "label_raw", "modality", "text"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter="\t", quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        for d in docs:
            writer.writerow(
                {
                    "id": d["id"],
                    "label": d["label"],
                    "label_raw": d["label_raw"],
                    "modality": d["modality"],
                    "text": d["text"],
                }
            )


# ----------------- Équilibrage -----------------

def rebalance_cap_docs(
    docs: List[Dict[str, Any]],
    cap: int,
    oversample: bool,
    offset: int,
) -> List[Dict[str, Any]]:
    """Reprise de la logique V2 pour cap_docs.

    - On regroupe par label.
    - Pour chaque label : on prend au plus `cap` docs (après rotation contrôlée).
    - Si oversample=True et qu'on a moins que cap, on duplique des docs jusqu'à cap.
    - On mélange à la fin.
    """
    by = defaultdict(list)
    for d in docs:
        by[d["label"]].append(d)

    rng = random.Random(offset)
    out: List[Dict[str, Any]] = []
    for lab, L in by.items():
        n = len(L)
        if n == 0:
            continue
        start = (hash(lab) + offset) % n
        rot = L[start:] + L[:start]
        take = min(cap, n)
        out.extend(rot[:take])
        if oversample and n < cap:
            need = cap - n
            for i in range(need):
                dup = dict(rng.choice(L))
                dup["id"] = f'{dup["id"]}#dup{i+1}'
                out.append(dup)
    rng.shuffle(out)
    return out


def rebalance_cap_tokens(
    docs: List[Dict[str, Any]],
    cap_tokens: int,
    offset: int,
) -> List[Dict[str, Any]]:
    """Reprise V2 pour cap_tokens (par label).

    On sélectionne, par label, suffisamment de docs (et de duplications)
    pour atteindre ~cap_tokens tokens.
    """
    by = defaultdict(list)
    for d in docs:
        by[d["label"]].append(d)

    out: List[Dict[str, Any]] = []
    for lab, L in by.items():
        n = len(L)
        if n == 0:
            continue
        start = (hash(lab) + offset) % n
        rot = L[start:] + L[:start]
        tot = 0
        buf: List[Dict[str, Any]] = []
        # Première passe : on prend tant qu'on n'a pas atteint cap_tokens
        for d in rot:
            if tot >= cap_tokens:
                break
            buf.append(d)
            tot += int(d.get("tokens") or 0)
        # Si on n'a toujours pas assez de tokens, on duplique
        i = 0
        while tot < cap_tokens and i < max(n, 1):
            dd = rot[i % n]
            dup = dict(dd)
            dup["id"] = f'{dd["id"]}#dup{i+1}'
            buf.append(dup)
            tot += int(dd.get("tokens") or 0)
            i += 1
        out.extend(buf)
    random.Random(offset).shuffle(out)
    return out


def rebalance_alpha_total(
    docs: List[Dict[str, Any]],
    alpha: float,
    total: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Reprise fidèle de rebalance_alpha_total de V2.

    - On calcule cnt(label).
    - Poids w(label) = (cnt(label) ** alpha).
    - Probabilité p(label) = w(label) / sum(w).
    - Quotas entiers par arrondi, ajustés pour total.
    - Échantillonnage avec duplication si besoin.
    """
    by = defaultdict(list)
    for d in docs:
        by[d["label"]].append(d)

    # comptages
    cnt = {lab: len(L) for lab, L in by.items()}
    w = {lab: (c ** alpha) for lab, c in cnt.items() if c > 0}
    z = sum(w.values()) or 1.0
    probs = {lab: wv / z for lab, wv in w.items()}
    quotas = {lab: max(1, int(round(total * p))) for lab, p in probs.items()}

    # Ajuster quotas pour que la somme soit exactement total
    diff = total - sum(quotas.values())
    labs = list(quotas.keys())
    i = 0
    while diff != 0 and labs:
        quotas[labs[i % len(labs)]] += 1 if diff > 0 else -1
        i += 1
        diff = total - sum(quotas.values())

    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for lab, L in by.items():
        k = quotas.get(lab, 0)
        if k <= 0 or not L:
            continue
        if k <= len(L):
            out.extend(rng.sample(L, k))
        else:
            out.extend(L)
            for j in range(k - len(L)):
                dup = dict(rng.choice(L))
                dup["id"] = f'{dup["id"]}#dup{j+1}'
                out.append(dup)
    rng.shuffle(out)
    return out


def apply_balance(
    docs: List[Dict[str, Any]],
    params: Dict[str, Any],
    label_counts: Counter,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """
    Appliquer la stratégie d'équilibrage configurée.

    Stratégies supportées :
      - "none"         : pas de modification
      - "cap_docs"     : cap sur le nombre de documents par label
      - "cap_tokens"   : cap sur le nombre de tokens par label
      - "alpha_total"  : quotas par label (version V2)
      - "class_weights": pas de resampling, calcul de poids de classe
    """
    strategy = params.get("balance_strategy", "none")
    balance_cfg = params.get("balance_cfg", {})
    preset_name = params.get("balance_preset")
    presets = balance_cfg.get("presets", {})
    preset = presets.get(preset_name, {}) if preset_name else {}
    seed = int(params.get("seed", 42))

    # No-op
    if strategy == "none" or not strategy:
        return docs, label_counts

    # Cap sur le nombre de documents par label
    if strategy == "cap_docs":
        cap_per_label = preset.get("cap_per_label")
        if cap_per_label is None:
            # Pas de preset valide → on ne touche pas aux données
            return docs, label_counts
        new_docs, new_counts = rebalance_cap_docs(
            docs=docs,
            label_counts=label_counts,
            cap_per_label=cap_per_label,
            seed=seed,
        )
        return new_docs, new_counts

    # Cap sur le nombre de tokens par label
    if strategy == "cap_tokens":
        cap_tokens_per_label = preset.get("cap_tokens_per_label")
        if cap_tokens_per_label is None:
            return docs, label_counts
        tokenizer_name = params.get("tokenizer", "split")
        new_docs, new_counts = rebalance_cap_tokens(
            docs=docs,
            label_counts=label_counts,
            cap_tokens_per_label=cap_tokens_per_label,
            tokenizer_name=tokenizer_name,
            seed=seed,
        )
        return new_docs, new_counts

    # Stratégie alpha_total (version V2 : quotas par label + oversampling contrôlé)
    if strategy == "alpha_total":
        alpha = float(preset.get("alpha", 1.0))
        total_docs = preset.get("total_docs")
        offset = int(preset.get("offset", 0))
        if total_docs is None:
            total_docs = len(docs)
        new_docs, new_counts = rebalance_alpha_total(
            docs=docs,
            label_counts=label_counts,
            alpha=alpha,
            total_docs=int(total_docs),
            offset=offset,
            seed=seed,
        )
        return new_docs, new_counts

    # Stratégie par poids de classe uniquement : pas de resampling
    if strategy == "class_weights":
        # On ne modifie pas docs : on calcule juste des poids à utiliser plus tard
        label_weights = compute_class_weights_from_counts(label_counts)
        # On les stocke dans params pour que core_train puisse les récupérer
        params["class_weights"] = label_weights
        return docs, label_counts

    # Stratégie inconnue → no-op
    return docs, label_counts




# ----------------- Formats (TSV -> formats modèles) -----------------


def build_formats(params: Dict[str, Any], meta_view: Dict[str, Any]) -> None:
    """
    Construire les formats pour les familles (spaCy, sklearn, HF, check, ...).

    V4-v1 :
      - spaCy : DocBin train/job
      - sklearn/hf/check : utilisent directement les TSV (train.tsv/job.tsv),
        mais sont référencés dans meta_formats.json.
    """
    families = params.get("families", []) or []
    corpus_id = params.get("corpus_id", params["corpus"].get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")

    interim_dir = Path("data") / "interim" / corpus_id / view
    processed_root = Path("data") / "processed" / corpus_id / view
    processed_root.mkdir(parents=True, exist_ok=True)

    train_tsv = interim_dir / "train.tsv"
    job_tsv = interim_dir / "job.tsv"

    formats_meta: Dict[str, Any] = {
        # "profile": params.get("profile"),
        "corpus_id": corpus_id,
        "view": view,
        "pipeline_version": PIPELINE_VERSION,
        "families": {},
    }
    # ---- spaCy : construction des DocBin ----
    if "spacy" in families:
        # S'assurer que la clé "families" existe dans formats_meta
        formats_meta.setdefault("families", {})

        spacy_dir = processed_dir / "spacy"
        spacy_dir.mkdir(parents=True, exist_ok=True)

        hardware = params.get("hardware", {}) or {}
        shard_docs = int(hardware.get("spacy_shard_docs") or 0)
        # NOTE : max_train_docs_spacy pourra être utilisé plus tard dans train_spacy_model
        lang = get_default_spacy_lang(params)

    # ---- spaCy : construction des DocBin ----
    if "spacy" in families:
        # S'assurer que la clé "families" existe dans formats_meta
        formats_meta.setdefault("families", {})

        spacy_dir = processed_dir / "spacy"
        spacy_dir.mkdir(parents=True, exist_ok=True)

        hardware = params.get("hardware", {}) or {}
        shard_docs = int(hardware.get("spacy_shard_docs") or 0)
        # NOTE : max_train_docs_spacy pourra être utilisé plus tard dans train_spacy_model
        lang = get_default_spacy_lang(params)

        if spacy is None:
            print("[core_prepare] WARNING: spaCy indisponible, impossible de construire les DocBin.")
        else:
            from spacy.tokens import DocBin

            def build_docbins(tsv_path: Path, prefix: str) -> Tuple[set, List[str], int]:
                """Construire un ou plusieurs DocBin à partir d'un TSV.

                Retourne (labels_set, chemins_relatifs, nb_docs).
                """
                labels_set: set = set()
                docbin_paths: List[str] = []
                total_docs = 0
                shard_idx = 0

                nlp = spacy.blank(lang)
                # On ne monte pas de pipeline complète ici, seulement le tokenizer
                db = DocBin()
                docs_in_current = 0

                with tsv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        text = row.get("text", "")
                        label = row.get("label")
                        if label is None:
                            continue
                        labels_set.add(label)
                        doc = nlp.make_doc(text)
                        # TextCat exclusif : une seule étiquette active
                        doc.cats = {label: 1.0}
                        db.add(doc)
                        docs_in_current += 1
                        total_docs += 1

                        # Sharding éventuel
                        if shard_docs and docs_in_current >= shard_docs:
                            out_path = spacy_dir / f"{prefix}_{shard_idx:03d}.spacy"
                            db.to_disk(out_path)
                            docbin_paths.append(out_path.name)
                            shard_idx += 1
                            db = DocBin()
                            docs_in_current = 0

                # Dernier shard / fichier unique
                if docs_in_current > 0:
                    if shard_docs:
                        out_path = spacy_dir / f"{prefix}_{shard_idx:03d}.spacy"
                    else:
                        out_path = spacy_dir / f"{prefix}.spacy"
                    db.to_disk(out_path)
                    docbin_paths.append(out_path.name)

                return labels_set, docbin_paths, total_docs

            labels_train, train_paths, n_train = build_docbins(train_tsv, "train")
            labels_job, job_paths, n_job = build_docbins(job_tsv, "job")
            labels_set = sorted(labels_train.union(labels_job))

            # Pour compatibilité : si pas de sharding, train_paths/job_paths ont 1 élément
            formats_meta["families"]["spacy"] = {
                "train_spacy": train_paths if shard_docs else (train_paths[0] if train_paths else None),
                "job_spacy": job_paths if shard_docs else (job_paths[0] if job_paths else None),
                "labels_set": labels_set,
                "lang": lang,
                "n_train_docs": n_train,
                "n_job_docs": n_job,
                "spacy_shard_docs": shard_docs,
            }



    # ---- sklearn / hf / check : on référence les TSV ----
    # (ils continueront d'utiliser train.tsv/job.tsv directement)
    for fam in ("sklearn", "hf", "check"):
        if fam in families:
            formats_meta["families"][fam] = {
                "source": "tsv",
                "train_tsv": str(train_tsv),
                "job_tsv": str(job_tsv) if job_tsv.exists() else None,
            }

    meta_proc_path = processed_root / "meta_formats.json"
    with meta_proc_path.open("w", encoding="utf-8") as f:
        json.dump(formats_meta, f, ensure_ascii=False, indent=2)

    print(f"[core_prepare] meta_formats.json écrit : {meta_proc_path}")


def build_spacy_formats(
    params: Dict[str, Any],
    train_tsv: Path,
    job_tsv: Path,
    processed_root: Path,
) -> Dict[str, Any]:
    """
    Construire les DocBin spaCy à partir de train.tsv/job.tsv.

    - data/processed/{corpus_id}/{view}/spacy/train.spacy
    - data/processed/{corpus_id}/{view}/spacy/job.spacy
    """
    try:
        import spacy
        from spacy.tokens import DocBin
    except ImportError:
        raise SystemExit("[core_prepare] spaCy n'est pas installé, impossible de construire les formats 'spacy'.")

    # Déterminer la langue à partir du premier modèle spaCy déclaré, sinon fallback "fr"
    models_cfg = params.get("models_cfg", {}).get("families", {}).get("spacy", {})
    models_spacy = params.get("models_spacy") or []
    lang = "fr"
    for mid in models_spacy:
        cfg = models_cfg.get(mid)
        if cfg and "lang" in cfg:
            lang = cfg["lang"]
            break

    nlp = spacy.blank(lang)

    def read_tsv(path: Path) -> Tuple[List[str], List[str]]:
        texts: List[str] = []
        labels: List[str] = []
        if not path.exists():
            return texts, labels
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

    train_texts, train_labels = read_tsv(train_tsv)
    job_texts, job_labels = read_tsv(job_tsv)

    if not train_texts:
        raise SystemExit("[core_prepare] Impossible de construire les formats spaCy : train.tsv vide ou introuvable.")

    labels_set = sorted(set(train_labels + job_labels))
    print(f"[core_prepare:spacy_formats] {len(train_texts)} docs train, {len(job_texts)} docs job, labels={labels_set}")

    spacy_dir = processed_root / "spacy"
    spacy_dir.mkdir(parents=True, exist_ok=True)

    # ---- DocBin train ----
    db_train = DocBin(store_user_data=True)
    for text, label in zip(train_texts, train_labels):
        doc = nlp.make_doc(text)
        cats = {lab: (lab == label) for lab in labels_set}
        doc.cats = cats
        db_train.add(doc)
    train_docbin_path = spacy_dir / "train.spacy"
    db_train.to_disk(train_docbin_path)

    # ---- DocBin job (si dispo) ----
    job_docbin_path = None
    if job_texts:
        db_job = DocBin(store_user_data=True)
        for text, label in zip(job_texts, job_labels):
            doc = nlp.make_doc(text)
            cats = {lab: (lab == label) for lab in labels_set}
            doc.cats = cats
            db_job.add(doc)
        job_docbin_path = spacy_dir / "job.spacy"
        db_job.to_disk(job_docbin_path)

    # Meta spécifique spaCy
    meta_spacy = {
        "lang": lang,
        "labels": labels_set,
        "n_train_docs": len(train_texts),
        "n_job_docs": len(job_texts),
        "train_docbin": str(train_docbin_path),
        "job_docbin": str(job_docbin_path) if job_docbin_path else None,
    }

    meta_spacy_path = spacy_dir / "meta_spacy_formats.json"
    with meta_spacy_path.open("w", encoding="utf-8") as f:
        json.dump(meta_spacy, f, ensure_ascii=False, indent=2)

    print(f"[core_prepare:spacy_formats] train.spacy/job.spacy écrits dans {spacy_dir}")

    return meta_spacy



# ----------------- main -----------------


def main() -> None:
    args = parse_args()
    params = resolve_profile_base(args.profile, args.override)

    if args.verbose:
        debug_print_params(params)

    meta_view = build_view(params, dry_run=args.dry_run)

    if not args.dry_run:
        build_formats(params, meta_view)


if __name__ == "__main__":
    main()
