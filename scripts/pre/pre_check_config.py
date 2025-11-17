# scripts/pre/pre_check_config.py

import argparse
import os
from typing import Any, Dict

from scripts.core.core_utils import (
    resolve_profile_base,
    load_yaml,
    load_label_map,
    debug_print_params,
)


def validate_models(params: Dict[str, Any]) -> None:
    models_cfg = params["models_cfg"]
    families = params.get("families", [])

    def check_model_list(family_key: str, list_key: str) -> None:
        model_ids = params.get(list_key, []) or []
        family_cfg = models_cfg.get("families", {}).get(family_key, {})
        for mid in model_ids:
            if mid not in family_cfg:
                raise SystemExit(
                    f"[config] Modèle '{mid}' non trouvé dans models.yml (famille '{family_key}')"
                )

    if "spacy" in families:
        check_model_list("spacy", "models_spacy")
    if "sklearn" in families:
        check_model_list("sklearn", "models_sklearn")
    if "hf" in families:
        check_model_list("hf", "models_hf")
    if "check" in families:
        # on peut ne rien faire ou vérifier plus tard
        pass


def validate_label_map(params: Dict[str, Any]) -> None:
    path = params.get("label_map")
    if not path:
        print("[config] WARNING: label_map non défini dans le profil")
        return
    if not os.path.exists(path):
        raise SystemExit(f"[config] label_map introuvable: {path}")
    mapping = load_label_map(path)
    if not mapping:
        print(f"[config] WARNING: label_map '{path}' chargé mais mapping vide (valeurs vides ?)")


def validate_corpus_and_labels(params: Dict[str, Any]) -> None:
    """Valider que le corpus et les label_maps référencés existent et sont chargeables."""
    corpus = params.get("corpus", {})
    corpus_path = corpus.get("corpus_path")
    if not corpus_path:
        raise SystemExit("[config] corpus.corpus_path manquant dans les paramètres résolus.")

    if not os.path.exists(corpus_path):
        print(f"[config] WARNING: corpus_path introuvable sur disque : {corpus_path}")

    label_map_path = params.get("label_map")
    if label_map_path:
        if not os.path.exists(label_map_path):
            raise SystemExit(f"[config] label_map inexistant : {label_map_path}")
        try:
            load_label_map(label_map_path)
        except Exception as e:
            raise SystemExit(f"[config] Impossible de charger label_map={label_map_path} : {e}")

    families = params.get("families") or []
    known_families = {"spacy", "sklearn", "hf", "check"}
    for fam in families:
        if fam not in known_families:
            raise SystemExit(f"[config] Famille inconnue dans profil : {fam!r} (attendu dans {sorted(known_families)})")

    # Vérifier que les modèles référencés existent dans models.yml
    models_cfg = params.get("models_cfg", {}).get("families", {})
    def _check_models(key: str, family: str) -> None:
        model_ids = params.get(key) or []
        for mid in model_ids:
            if mid not in models_cfg.get(family, {}):
                raise SystemExit(
                    f"[config] Modèle '{mid}' référencé dans {key} "
                    f"mais absent de configs/common/models.yml pour la famille '{family}'."
                )

    _check_models("models_spacy", "spacy")
    _check_models("models_sklearn", "sklearn")
    _check_models("models_hf", "hf")
    # Pour 'check', on tolère pour l'instant un pseudo-modèle implicite 'check_default'


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pré-check d'un profil V4 (cohérence configs)"
    )
    ap.add_argument("--profile", required=True, help="Nom du profil (sans .yml)")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config (clé=valeur, ex: train_prop=0.7)",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Afficher les params résolus"
    )
    args = ap.parse_args()

    params = resolve_profile_base(args.profile, args.override)

    # Validations
    validate_label_map(params)
    validate_models(params)
    validate_corpus_and_labels(params)


        # 1) Existence des templates spaCy référencés
    models = cfg_all["models_cfg"]["families"].get("spacy", {})
    missing = []
    for mid, mc in models.items():
        tpl = mc.get("config_template")
        if tpl and not Path(tpl).exists():
            missing.append((mid, tpl))
    if missing:
        for mid, tpl in missing:
            print(f"[pre_check] MISSING spaCy config_template for model '{mid}': {tpl}")
        raise SystemExit("[pre_check] Missing spaCy config templates.")

    # 2) Cohérence FAMILIES demandées vs models.yml
    families_req = set(params.get("families", []))
    known_fams = set(cfg_all["models_cfg"]["families"].keys())
    unknown = families_req - known_fams
    if unknown:
        raise SystemExit(f"[pre_check] Unknown families requested: {sorted(unknown)}")

    # 3) Hardware preset connu
    hp = params.get("hardware_preset", "small")
    if hp not in cfg_all["hardware_cfg"].get("presets", {}):
        raise SystemExit(f"[pre_check] Unknown hardware_preset: {hp}")



    # Vérification hardware minimale
    hw = params.get("hardware", {})
    if not hw:
        print("[config] WARNING: pas de hardware_preset appliqué")
    else:
        if hw.get("ram_gb", 0) <= 0:
            print("[config] WARNING: ram_gb non réaliste")
        if hw.get("max_procs", 0) <= 0:
            print("[config] WARNING: max_procs non réaliste")

    if args.verbose:
        debug_print_params(params)

    print(f"[OK] Profil '{args.profile}' validé.")


if __name__ == "__main__":
    main()
