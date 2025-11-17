# scripts/core/core_utils.py

import argparse
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

import yaml

COMMON_DIR = os.path.join("configs", "common")
PIPELINE_VERSION = "4.0.0-dev"

# ---------- Utils de base ----------

def load_yaml(path: str) -> Dict[str, Any]:
    """Charger un YAML en dict, avec un message d'erreur clair."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML introuvable: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Fusion récursive de dictionnaires (base modifié in-place, retourné)."""
    for k, v in updates.items():
        if (
            isinstance(v, dict)
            and k in base
            and isinstance(base[k], dict)
        ):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_override(raw: str) -> (List[str], Any):
    """
    Parse une override "a.b.c=val" -> (["a","b","c"], "val")
    On laisse la responsabilité de caster au code qui applique
    (int, float, bool, etc.) si besoin.
    """
    if "=" not in raw:
        raise ValueError(f"Override invalide (pas de '='): {raw}")
    key, value = raw.split("=", 1)
    path = key.split(".")
    return path, value


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Appliquer une liste de 'key=value' sur un dict (nested)."""
    cfg = deepcopy(config)
    for raw in overrides:
        path, value = parse_override(raw)
        # Tentative de cast simple
        if isinstance(value, str) and value.lower() in ("true", "false"):
            cast_val: Any = value.lower() == "true"
        else:
            try:
                cast_val = int(value)
            except (ValueError, TypeError):
                try:
                    cast_val = float(value)
                except (ValueError, TypeError):
                    cast_val = value

        d: Dict[str, Any] = cfg
        for key in path[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}
            d = d[key]
        d[path[-1]] = cast_val
    return cfg


# ---------- Label maps ----------

def load_label_map(path: str) -> Dict[str, str]:
    """
    Charger un fichier de mapping de labels / acteurs.
    Supporte deux formes :
      1) {'mapping': {clé: valeur, ...}}
      2) {clé: valeur, ...} directement (ex: YAML produit par make_ideology_skeleton
         puis rempli manuellement).
    Retourne un dict {clé: str(valeur_non_vide)}.
    """
    raw = load_yaml(path)
    if "mapping" in raw and isinstance(raw["mapping"], dict):
        mapping = raw["mapping"]
    else:
        mapping = raw

    result: Dict[str, str] = {}
    for k, v in mapping.items():
        if v is None:
            continue
        v_str = str(v).strip()
        if not v_str:
            continue
        result[str(k)] = v_str
    return result


# ---------- Résolution de profil ----------

def resolve_profile_base(profile_name: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Charge profil + YAML communs (corpora/balance/hardware/models) et construit 'params'.
    """
    if overrides is None:
        overrides = []

    profile_path = os.path.join("configs", "profiles", f"{profile_name}.yml")
    profile_cfg = load_yaml(profile_path)

    corpora_cfg  = load_yaml(os.path.join(COMMON_DIR, "corpora.yml"))
    balance_cfg  = load_yaml(os.path.join(COMMON_DIR, "balance.yml"))
    hardware_cfg = load_yaml(os.path.join(COMMON_DIR, "hardware.yml"))
    models_cfg   = load_yaml(os.path.join(COMMON_DIR, "models.yml"))

    corpus_id = profile_cfg.get("corpus_id")
    if corpus_id not in corpora_cfg:
        raise SystemExit(f"[config] corpus_id '{corpus_id}' non défini dans common/corpora.yml")

    params: Dict[str, Any] = {
        "profile": profile_cfg.get("profile", profile_name),
        "description": profile_cfg.get("description", ""),
        "corpus": corpora_cfg[corpus_id],
        "profile_raw": profile_cfg,
        "balance_cfg": balance_cfg,
        "hardware_cfg": hardware_cfg,
        "models_cfg": models_cfg,
        "seed": profile_cfg.get("seed", 42),
    }

    # Hardware preset (une seule fois, sans doublon)
    hardware_preset  = profile_cfg.get("hardware_preset", "small")
    hardware_presets = hardware_cfg.get("presets", {})
    params["hardware_preset"] = hardware_preset
    params["hardware"] = hardware_presets.get(hardware_preset, {})

    # Champs simples copiés depuis le profil
    simple_keys = [
        "corpus_id", "view", "modality",
        "label_field", "label_map",
        "train_prop", "min_chars", "max_tokens",
        "tokenizer", "seed",
        "balance_strategy", "balance_preset", "balance_mode",
        "dedup_on",
        "families",
        "models_spacy", "models_sklearn", "models_hf", "models_check",
        "hardware_preset", "debug_mode",
    ]
    for k in simple_keys:
        if k in profile_cfg:
            params[k] = profile_cfg[k]

    # Overrides CLI au niveau 'params'
    params["pipeline_version"] = PIPELINE_VERSION
    params = apply_overrides(params, overrides)

    # Alias de confort : balance_mode -> balance_strategy
    bm = (params.get("balance_mode") or "").strip().lower()
    if bm == "weights":
        params["balance_strategy"] = "class_weights"
    elif bm == "oversample":
        params.setdefault("balance_strategy", "cap_docs")

    return params



# --- Reproducibilité globale (optionnelle) ---

def apply_global_seed(seed_val) -> bool:
    """
    Fixe la seed pour random/numpy/torch/spaCy si seed_val est un entier.
    Si seed_val est None/'none'/'null' ou <0 -> on N'APPLIQUE PAS de seed (comportement optionnel).
    Retourne True si une seed a été appliquée.
    """
    try:
        if seed_val is None:
            return False
        if isinstance(seed_val, str) and seed_val.strip().lower() in {"none", "null", ""}:
            return False
        seed = int(seed_val)
        if seed < 0:
            return False
    except Exception:
        return False

    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        # spaCy >=3
        import spacy.util as spacy_util
        spacy_util.fix_random_seed(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    return True




def debug_print_params(params: Dict[str, Any]) -> None:
    """Petit helper pour inspecter les params (pour debug)."""
    import pprint
    print("=== PARAMS V4 RÉSOLUS ===")
    pprint.pprint(params)
