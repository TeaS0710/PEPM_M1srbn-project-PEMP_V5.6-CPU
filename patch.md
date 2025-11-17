1. Une **checklist exhaustive** des patchs (avec regroupement logique).
2. Un **plan conceptuel** (comment organiser la complexité sans faire un monstre opaque).
3. Un **plan technique détaillé** (brique par brique, fichiers, tâches).
4. Une liste d’**erreurs conceptuelles à éviter**.

---

## 1. Checklist exhaustive des patchs V4

Je regroupe tout ce qu’il y a à faire (y compris briques transverses) en blocs.
Tu peux presque la coller dans un `dev_V4_todo.md` avec cases à cocher.

### Bloc 1 – `core_prepare` & data layer

1. **Équilibrage avancé V2-compliant**

   * [ ] Implémenter / vérifier `apply_balance` avec 5 stratégies :

     * `none`, `cap_docs`, `cap_tokens`, `alpha_total`, `class_weights`.
   * [ ] Lire tous les presets dans `configs/common/balance.yml` :

     * `cap_per_label`, `cap_tokens_per_label`, `alpha`, `total_docs`, `offset`, `oversample` éventuel…
   * [ ] Implémenter / porter fidèlement :

     * `rebalance_cap_docs(...)`
     * `rebalance_cap_tokens(...)`
     * `rebalance_alpha_total(...)` (version V2 exacte)
     * `compute_class_weights_from_counts(...)`
   * [ ] Gérer explicitement les docs sans label après `label_map` (drop + stats).
   * [ ] Enregistrer dans `meta_view.json` :

     * `balance_strategy`, `balance_preset`
     * `label_counts_before`, `label_counts_after`
     * pour `class_weights`: `label_weights`.

2. **Multi-tokenizer & `max_tokens` réel**

   * [ ] Ajouter `tokenizer` dans les profils (`split` / `spacy_xx`).
   * [ ] Propager `tokenizer` dans `resolve_profile_base` (`params["tokenizer"]`).
   * [ ] Implémenter helpers :

     * `get_spacy_xx(lang)` (cache `spacy.blank(lang)`).
     * `count_tokens(text, tokenizer_name)` :

       * `spacy_xx` → `len(nlp.make_doc(text))`
       * fallback → `len(text.split())`.
   * [ ] Utiliser `count_tokens` dans `build_view()` pour :

     * filtrer `max_tokens`
     * stocker éventuellement `doc["tokens"]`.
   * [ ] Ajouter `tokenizer` et stats de tokens dans `meta_view.json`.

3. **DocBin + sharding spaCy**

   * [ ] Ajouter dans `configs/common/hardware.yml` :

     * `spacy_shard_docs` par preset (`small`, `lab`, …).
   * [ ] Dans `core_prepare.build_formats`, pour la famille `spacy` :

     * [ ] Créer un helper `build_docbins(tsv_path, prefix, shard_docs)` qui :

       * lit les TSV,
       * construit un ou plusieurs `DocBin` :

         * soit `train.spacy` / `job.spacy`,
         * soit `train_000.spacy`, `train_001.spacy`, …
     * [ ] Remplir `meta_formats["families"]["spacy"]` avec :

       * `train_spacy`: liste de fichiers ou chaîne,
       * `job_spacy`: idem,
       * `n_train_docs`, `n_job_docs`,
       * `spacy_shard_docs`.

---

### Bloc 2 – Entraînement modèles (spaCy / sklearn / HF / check)

4. **spaCy “config-first” (BOW/CNN)**

   * [ ] Créer des templates dans `configs/spacy/` :

     * `textcat_bow_base.cfg`
     * `textcat_cnn_base.cfg`
     * avec :

       * un composant `textcat`,
       * des placeholders pour dropout, epochs, etc.,
       * `paths.train`, `paths.dev` sur DocBin (à override).
   * [ ] Étendre `configs/common/models.yml` (famille `spacy`) :

     * pour chaque modèle :

       * `arch: "bow" | "cnn"`
       * `config_template: "configs/spacy/..."`,
       * `epochs`, `dropout`, autres hyperparams.
   * [ ] Dans `train_spacy_model(params, model_id)` :

     * [ ] si `config_template` défini :

       * charger le `.cfg`,
       * override `paths.train` / `paths.dev` à partir de `meta_formats`,
       * override hyperparams (dropout, epochs…) depuis YAML,
       * lancer training façon `spacy.cli.train` programmatique,
       * sauvegarder pipeline dans `model_dir`.
     * [ ] sinon : fallback “training maison” actuel.
   * [ ] Dans `meta_model.json` :

     * `arch`, `config_template`, `train_mode` (`config` / `legacy`),
     * `n_train_docs`.

5. **sklearn : intégration hardware & meta**

   * [ ] Dans `hardware.yml` : `max_train_docs_sklearn`.
   * [ ] Dans `train_sklearn_model` :

     * après chargement du dataset, si `len(texts) > max_train_docs_sklearn` :

       * couper ou sous-échantillonner avec seed fixe.
     * stocker dans `meta_model.json` :

       * `n_train_docs_raw`, `n_train_docs_effective`,
       * `vectorizer_class`, `estimator_class`,
       * hyperparams (`vectorizer.params`, `estimator.params`),
       * info sur la stratégie de réduction (`"truncate_head"` / `"sample"`).
   * [ ] Vérifier que les `class_weights` sont bien portés côté sklearn quand pertinent (ex. `class_weight="balanced"` pour certains modèles).

6. **HF : training générique (CPU)**

   * [ ] Dans `models.yml` (famille `hf`) définir pour chaque modèle :

     * `desc`
     * `model_name` (HF hub)
     * `tokenizer_class` (via import dynamique)
     * `model_class`
     * `max_length`
     * `trainer_params` (learning_rate, batch_size, epochs, warmup_ratio, etc.)
     * `use_class_weights` (true/false)
   * [ ] Dans `train_hf_model(params, model_id)` :

     * [ ] charger `train.tsv` via `load_tsv_dataset`,
     * [ ] construire `label2id`, `id2label`,
     * [ ] importer `tokenizer_class`, `model_class` via `importlib`,
     * [ ] créer un Dataset HF (`__getitem__` → encodage + label),
     * [ ] créer `TrainingArguments` à partir de `trainer_params` + hardware preset,
     * [ ] entraîner via `Trainer`,
     * [ ] sauvegarder modèle + tokenizer dans `model_dir`,
     * [ ] écrire `meta_model.json` avec :

       * `hf_model_name`, `label2id`, `id2label`,
       * `trainer_params`, `n_train_docs_raw`, `n_train_docs_effective`.

7. **HF : eval générique**

   * [ ] Dans `eval_hf_model(params, model_id)` :

     * [ ] charger `model_dir` + tokenizer + `meta_model`,
     * [ ] charger `job.tsv` (ou fallback `train.tsv` avec warning),
     * [ ] construire Dataset d’éval HF,
     * [ ] faire passer les batchs dans le modèle,
     * [ ] récupérer `argmax(logits)` → ids → labels via `id2label`,
     * [ ] calculer `compute_basic_metrics`,
     * [ ] sauvegarder via `save_eval_outputs` (`metrics.json`, `classification_report.txt`, `meta_eval.json`).

8. **HF + `class_weights`**

   * [ ] Assurer que `core_prepare` génère bien les `label_weights` pour une stratégie `class_weights`.
   * [ ] Dans `train_hf_model` :

     * si `use_class_weights` :

       * soit récupérer `label_weights` depuis `meta_view`,
       * soit les recalculer à partir de `y_train`,
       * définir un `compute_loss` custom dans un Trainer subclass (pondérer la loss par classe).
   * [ ] Logguer dans `meta_model.json` :

     * `class_weights_used: true/false`,
     * les poids de chaque label.

9. **famille `check`**

   * [ ] Vérifier que `train_check_model` et `eval_check_model` :

     * ne dépendent d’aucun format spécifique,
     * produisent bien `meta_model.json` et `meta_eval.json` minimalistes,
     * sont toujours appelés quand `families` contient `check`.

---

### Bloc 3 – Évaluation & reports

10. **Éval spaCy / sklearn / HF alignée**

* [ ] Harmoniser `eval_spacy_model`, `eval_sklearn_model`, `eval_hf_model` :

  * même signature, même structure de `meta_eval`,
  * même usage de `load_job_tsv` / `maybe_debug_subsample_eval`.
* [ ] `compute_basic_metrics` :

  * `accuracy`, `f1_macro`, `classification_report` (str),
  * éventuellement F1 par classe dans la meta.
* [ ] `save_eval_outputs` :

  * `metrics.json`,
  * `classification_report.txt`,
  * `meta_eval.json` (avec `profile`, `corpus`, `view`, `model_id`, `family`).

---

### Bloc 4 – Hardware / RAM / BLAS

11. **Presets hardware & limites**

* [ ] Compléter `configs/common/hardware.yml` :

  * par preset :

    * `blas_threads`,
    * `spacy_shard_docs`,
    * `max_train_docs_spacy`,
    * `max_train_docs_sklearn`,
    * `max_train_docs_hf`.
* [ ] Dans `core_utils` :

  * s’assurer que `resolve_profile_base` construit bien `params["hardware"]` à partir du preset.
* [ ] Dans `core_train` :

  * `set_blas_threads` pour tous les trainings,
  * appliquer les `max_train_docs_*` dans chaque famille,
  * logguer dans `meta_model` les docs raw vs effective.

---

### Bloc 5 – Config system & overrides

12. **Configs YAML complètes et cohérentes**

* [ ] `configs/common/corpora.yml` :

  * déclarer tous les corpus actuels (`web1`, `web2`, …),
  * préciser chemins TEI, patterns TEI (`<term type="ideology">`, `crawl`, `modality`),
  * prévoir clés placeholder pour futurs corpus ASR/GOLD (`asr1`, `gold1`).
* [ ] `configs/common/balance.yml` :

  * définir les presets d’équilibrage,
  * expliquer dans un petit commentaire l’intention de chaque preset.
* [ ] `configs/common/models.yml` :

  * familles `spacy`, `sklearn`, `hf`, `check`,
  * pour chaque modèle : ID stable, desc, hyperparams.
* [ ] `configs/common/hardware.yml` (cf. ci-dessus).
* [ ] `configs/label_maps/*.yml` :

  * label_map pour idéologie, crawl, etc.
* [ ] `configs/profiles/*.yml` :

  * `ideo_quick`, `ideo_full`, `crawl_quick`, `crawl_full`, `check_only`, `custom`.

13. **`resolve_profile_base` & overrides**

* [ ] S’assurer que `resolve_profile_base` :

  * charge profil + corpora + balance + hardware + models,
  * remplit `params` avec :

    * `corpus`, `view`, `label_field`, `families`, `tokenizer`, `balance`, `hardware`, `models`, `seed`, etc.,
    * `pipeline_version`.
* [ ] Implémenter/terminer `apply_overrides` :

  * format `key1.key2=val`,
  * parsing de `val` en bool/int/float/str,
  * support des listes simples (ex: `families=[spacy,sklearn]` ou `families=spacy,sklearn`).
* [ ] Documenter la surface officielle de `OVERRIDES` (cf. bloc J plus bas).

14. **pre_check_config**

* [ ] Renforcer `scripts/pre/pre_check_config.py` pour vérifier :

  * existence du profil,
  * modèle(s) déclaré(s) existent dans `models.yml`,
  * corpus déclaré existe dans `corpora.yml`,
  * fichiers TEI et chemins existent,
  * équilibre, hardware, familles cohérents,
  * familles demandées supportées par ce profil (par ex. ne pas lancer HF si pas de `models.hf` définis).
* [ ] Rendre `pre_check_config` obligatoire avant `core_prepare` dans les cibles Makefile principales.

---

### Bloc 6 – Reproductibilité & logging

15. **Seed global & reproductibilité**

* [ ] Ajouter un champ `seed` dans tous les profils (ou valeur par défaut globale).
* [ ] Dans `resolve_profile_base` :

  * `params["seed"] = profile.get("seed", 42)`.
* [ ] Dans chaque `core_*` :

  * au début, fixer :

    ```python
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    try:
        import torch
        torch.manual_seed(params["seed"])
    except ImportError:
        pass
    ```
* [ ] Utiliser `params["seed"]` partout où il y a un tirage aléatoire (shuffle, sampling).
* [ ] Inclure `seed` dans tous les fichiers meta (`meta_view`, `meta_formats`, `meta_model`, `meta_eval`).

16. **Logging unifié**

* [ ] Créer un helper `get_logger(name, level)` (dans `core_utils` ou `logging_utils`).
* [ ] Remplacer les `print()` dans `core_prepare`, `core_train`, `core_evaluate` par :

  * `logger.info/warning/error`.
* [ ] Ajouter un param `log_level` dans les profils (ou défaut INFO).
* [ ] Optionnel : permettre d’écrire le log dans un fichier par run (`logs/PROFILE_TIMESTAMP.log`).

---

### Bloc 7 – Meta & provenance de run

17. **Schéma des métadonnées**

* [ ] Définir un schéma pour :

  * `meta_view.json` (niveau “view”),
  * `meta_formats.json` (formats générés),
  * `meta_model.json`,
  * `meta_eval.json`.
* [ ] Ajouter un champ :

  * `"schema_version": "view-1.0"`, `"model-1.0"`, etc.
  * `"pipeline_version"` (+ éventuellement `git_commit`).
* [ ] Assurer que chaque meta contient au minimum :

  * identifiants : corpus, view, profile, families,
  * seed, hardware_preset, tokenizer,
  * counts docs, labels, etc.
* [ ] Rédiger `docs/meta_schema_V4.md` qui décrit ces structures.

---

### Bloc 8 – Tests d’intégration & samples

18. **Mini-corpus de test**

* [ ] Créer un mini corpus TEI (ou TSV) dans `data/sample/` :

  * quelques documents, 2–3 labels.
* [ ] Créer un profil `sample_quick` ou réutiliser `ideo_quick` mais avec corpus=`sample`.

19. **Cibles Makefile de test**

* [ ] Ajouter dans le `Makefile` :

  * `make sample_prepare PROFILE=sample_quick`,
  * `make sample_train PROFILE=sample_quick`,
  * `make sample_eval PROFILE=sample_quick`,
  * ou un `make sample_pipeline PROFILE=sample_quick` (prepare+train+eval).
* [ ] S’assurer que cette pipeline tourne vite et entièrement.

20. **Tests de régression sur l’équilibrage**

* [ ] Créer un micro TSV (10–20 lignes) avec distribution de labels connue.
* [ ] Écrire un petit script `scripts/tests/test_balance_regression.py` ou équivalent qui :

  * appelle `apply_balance` avec chaque stratégie,
  * compare le résultat à un JSON de référence (produit depuis V2),
  * raise si divergence.
* [ ] Documenter ce test dans `docs/howto_tests_V4.md`.

---

### Bloc 9 – Multi-modalité future & ingestion TEI

21. **Préparer les corpus ASR/GOLD**

* [ ] Dans `corpora.yml`, ajouter la structure prévue pour :

  * `asr1`, `gold1`, même si pas encore utilisés.
* [ ] Définir les champs (meta) qui distinguent web / asr / gold :

  * type de source, encodage, champs TEI différents, etc.
* [ ] Vérifier que `core_prepare` :

  * ne fait **aucune hypothèse** idéologie-spécifique hormis `label_field` et `view`,
  * ne dépend pas d’un type unique de `<term>`.

22. **Scripts d’ingestion**

* [ ] Si tu réutilises `make_ideology_skeleton.py` / ajout de `<term type="modality">` :

  * s’assurer qu’ils sont rangés dans `scripts/pre/`,
  * qu’ils ne font pas partie du “core” mais bien de la phase ingestion / annotation.
* [ ] Documenter clairement la frontière :

  * ingestion = produire un TEI conforme (web, asr, gold),
  * core V4 = à partir du TEI conforme, pipeline identique.

---

### Bloc 10 – Makefile / CLI / surface officielle

23. **Surface Makefile**

* [ ] Geler les variables supportées :

  * `PROFILE`, `FAMILIES`, `OVERRIDES`, `HARDWARE`, `DEBUG`, etc.
* [ ] S’assurer que toutes les cibles V4 utilisent uniquement :

  * `pre_check_config`, `core_prepare`, `core_train`, `core_evaluate`,
  * plus quelques scripts post/experiments optionnels.
* [ ] Retirer les appels aux scripts V2/V3, s’il en reste.

24. **CLI Python optionnelle**

* [ ] Option : ajouter un `scripts/cli.py` ou `pepm/__main__.py` qui permet :

  * `python -m pepm.core.prepare --profile ideo_quick`,
  * `python -m pepm.core.train --profile ideo_quick`.
* [ ] Documenter que le Makefile est un **wrapper pratique**, la référence restant le couple configs+core.

---

### Bloc 11 – Doc & packaging

25. **Doc dev & howto**

* [ ] Finaliser `docs/dev_V4.md` (architecture, choix, héritage V1–V3).
* [ ] Compléter `docs/ref_V4_parameters.md` :

  * lister tous les paramètres :

    * profils, models, balance, hardware, overrides,
    * valeurs possibles, défauts, impacts.
* [ ] Rédiger `docs/howto_V4.md` :

  * “Ajouter un modèle spaCy”,
  * “Ajouter un modèle HF”,
  * “Ajouter un corpus”,
  * “Ajouter un profil d’expérience”.

26. **Packaging léger**

* [ ] (Optionnel mais propre) : structurer le code en pseudo-package :

  * `src/pepm/core/...`,
  * `src/pepm/pre/...`,
  * etc.
* [ ] Ajouter un `pyproject.toml` ou `setup.cfg` minimal.
* [ ] Documenter comment ajouter le projet au `PYTHONPATH`.

---

### Bloc 12 – Legacy & nettoyage

27. **Archiver V1/V2/V3**

* [ ] Déplacer les versions précédentes dans `archive/PEPM_V1`, `V2`, `V3` ou repos séparés.
* [ ] Ajouter une section “Historique des versions” dans `dev_V4.md` qui :

  * raconte V1.1 → V2 → V3 (échec) → V4,
  * indique où trouver les anciens scripts au besoin.
* [ ] S’assurer qu’aucun script V2/V3 n’est importé ni appelé par V4.

---

## 2. Plan conceptuel – organiser la richesse sans rendre V4 opaque

### 2.1 Les 6 axes de variabilité

Conceptuellement, tout ton système tourne autour de **6 axes** :

1. **Corpus** (`corpora.yml`)

   * web1 / web2 / (asr1 / gold1 plus tard),
   * champs TEI, chemins, langue, taille, etc.

2. **View / task**

   * `ideo_web`, `crawl_web`, etc.
   * associée à `label_field` (ideology, crawl, …).

3. **Famille de modèle**

   * `spacy`, `sklearn`, `hf`, `check`.

4. **Modèle concret**

   * `spacy_cnn_quick`, `tfidf_svm_base`, `camembert_base`, etc.
   * décrit uniquement dans `models.yml`.

5. **Stratégie d’équilibrage / sampling**

   * `none`, `cap_docs`, `cap_tokens`, `alpha_total`, `class_weights`.

6. **Preset hardware**

   * `small`, `lab`, etc.
   * contrôle shards, max docs, threads BLAS.

**Concept clé :**
V4 doit rester **combinatoire et paramétrable** **uniquement via configs**.
Le code core ne doit pas avoir de cas spéciaux pour des `model_id` ou `corpus_id` particuliers.

---

### 2.2 Frontières claires

* **Ingestion / annotation TEI** (web, asr, gold…)
  → scripts dans `scripts/pre/`
  → produisent un **TEI conforme** (balises, champs attendus).

* **Core V4** (prepare/train/evaluate)
  → scripts dans `scripts/core/`
  → à partir d’un TEI conforme + config YAML,
  → pipeline identique quel que soit le corpus ou la tâche.

* **Expériences / explorations**
  → `scripts/experiments/`, `scripts/analysis/`
  → consomment les outputs du core, ne redéfinissent pas la logique centrale.

---

### 2.3 Surface officielle : configs + Make/CLI

* L’**API** de V4 pour un utilisateur = :

  * choisir un **profil**,
  * éventuellement spécifier `FAMILIES` + `OVERRIDES` + `HARDWARE`,
  * lancer une commande type `make pipeline` ou `python -m pepm.core.*`.

* Tout le reste est considéré comme **interne** (impl détails des familles, etc.).

---

### 2.4 Méta & reproductibilité comme contrat

Chaque run doit :

* être **rejouable** (seed global, configs figées),
* laisser une **trace complète** (meta + logs) expliquant :

  * quoi (corpus, view, profil, modèles),
  * comment (balance, hardware, tokenizer, seeds),
  * avec quels résultats (metrics).

---

## 3. Plan technique – par briques

La checklist ci-dessus est déjà un plan technique.
En gros, l’ordre conseillé pour implémenter :

1. **Stabiliser `core_prepare`**
   (équilibrage V2-compliant, multi-tokenizer, DocBin/shards, meta_view).

2. **Stabiliser `core_train` & familles**

   * spaCy config-first, fallback legacy, sharding,
   * sklearn (RAM & meta),
   * HF (train, eval, class_weights).

3. **Aligner `core_evaluate`**
   (familles homogènes, meta_eval commune).

4. **Compléter hardware.yml & intégration**
   (max docs, shards, BLAS).

5. **Renforcer le système de config & `pre_check_config`**
   (YAML complets, overrides, vérifs).

6. **Ajouter seed global & logging unifié**
   (reproductibilité + logs).

7. **Définir et documenter le schéma des métadonnées**
   (view/formats/model/eval).

8. **Mettre en place les tests d’intégration & samples**
   (sample corpus, make sample_pipeline, tests d’équilibrage).

9. **Préparer la multi-modalité future au niveau de la config**
   (corpora.yml asr/gold, frontières ingestion/core).

10. **Geler l’interface Makefile / CLI et la doc**
    (howto, ref parameters, historique versions).

11. **Archiver V1/V2/V3**
    (et nettoyer les références dans V4).

---

## 4. Erreurs conceptuelles à éviter (compte tenu de ta vision)

1. **Réintroduire de la logique métier cachée dans le Makefile**

   * Le Makefile doit rester une **colle** et non un second cerveau.
   * Pas de “si PROFILE=machin alors changer tel hyperparam” codé en dur côté Make.

2. **Faire des cas spéciaux par modèle / corpus dans le code core**

   * Pas de `if model_id == "camembert_base"` dans `core_train`.
   * Tout doit passer par `models.yml` + familles génériques.

3. **Empiler des flags dans tous les sens sans doc**

   * Tu veux un système riche en paramètres, mais il faut :

     * des **niveaux** (profil > modèles > hardware > balance),
     * une doc de référence (`ref_V4_parameters.md`),
     * une surface d’override **claire**.

4. **Coupler ingestion (TEI) et core**

   * Si `core_prepare` commence à faire des choses spécifiques à “idéologie web”, ça tue la multi-modalité.
   * Toute la “saleté” d’origine (tags, heuristiques, nettoyage) doit rester dans des scripts `pre/`.

5. **Changer silencieusement le comportement de l’équilibrage par rapport à V2**

   * Tu veux pouvoir dire : “V4 reproduit V2 sur l’équilibrage quand on choisit tel preset”.
   * Donc : tests de régression obligatoires, et log clair.

6. **Ignorer la reproductibilité**

   * Si tu touches aux seeds, sampling, shuffles **sans les relier à un `seed` unique**, tu ne pourras plus expliquer pourquoi deux runs diffèrent.

7. **Laisser les métadonnées dériver sans schéma**

   * Si chaque refacto ajoute/supprime des champs meta sans tracking, tu vas perdre la capacité à analyser les runs a posteriori.
   * D’où l’importance du schéma versionné.

8. **Cacher la complexité en “rendant V4 étanche”**

   * Ton objectif n’est pas de faire une boîte noire, mais une usine à gaz **documentée** :

     * tout doit être paramétrable,
     * mais la logique et la structure doivent rester lisibles via :

       * doc dev,
       * schéma meta,
       * howto.

---

Si tu veux, prochain round on peut :

* soit prendre un bloc précis (ex. **équilibrage V2-compliant en V4**) et je t’écris le pseudo-code + structure YAML + exemple de `meta_view`,
* soit attaquer la **doc ref paramètres** en la structurant (par “axes de variabilité”).
