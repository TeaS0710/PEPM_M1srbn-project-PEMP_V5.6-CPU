Patchs restants (priorisés)
A) spaCy “config-first” (bloquant)

Créer les templates dans configs/spacy/ :

textcat_bow_base.cfg (Tok2Vec + TextCat bag-of-words)

textcat_cnn_base.cfg (TextCat CNN)

placeholders standard : paths.train, paths.dev, training.dropout, training.max_epochs, batch sizes, etc.

Étendre models.yml (famille spacy) :

Ajouter des entrées BOW (ex. spacy_bow_quick, spacy_bow_full) avec :

arch: bow

config_template: configs/spacy/textcat_bow_base.cfg

epochs, dropout, etc.

Garder les CNN existants mais corriger les chemins si besoin.

Adapter train_spacy_model :

Si config_template est présent :

charger la config (spacy.util.load_config),

override paths.train et paths.dev avec les DocBin générés par core_prepare (shardés ou non) — possible en concaténant plusieurs DocBin côté config (ou en générant un train.spacy unique si besoin),

override hyperparams (ex. training.max_epochs, training.dropout) depuis models.yml,

lancer l’entraînement programmatique (équivalent spacy.cli.train),

sauver le pipeline dans models/.../spacy/{model_id}.

Sinon, fallback legacy (ton code actuel).

Évaluation : eval_spacy_model est OK (lit les DocBin job*.spacy) — juste vérifier la compatibilité avec tes templates.

B) HF – poids de classes (rapide à corriger)

Corriger models.yml : déplacer use_class_weights au niveau du modèle (pas dans trainer_params).
ou modifier train_hf_model pour aller le lire depuis trainer_params.get("use_class_weights").
→ Ça réactive la CrossEntropyLoss pondérée déjà codée via WeightedTrainer.

C) Hardware & RAM

spaCy : appliquer max_train_docs_spacy (déjà dans le YAML) au moment de constituer le dataset d’entraînement spaCy (si tu ne veux pas le faire dans prepare).

Optionnel : add max_eval_docs_* pour couper l’évaluation sur grosses machines en debug.

D) Meta & checks

Figé du schéma meta (+ schema_version) :

meta_view.json : tokenizer, balance_strategy, balance_preset, label_counts_before/after, label_weights (si class_weights), seed, etc.

meta_formats.json : par famille, chemins (liste vs str), n_train_docs, n_job_docs, lang, spacy_shard_docs.

meta_model.json (toutes familles) : n_train_docs_raw/effective, params entraînement, label2id/id2label (HF), etc.

meta_eval.json : n_eval_docs, timestamp, modèle, famille, profil, etc.

pre_check_config.py :

vérifier l’existence des config_template spaCy,

valider cohérence familles demandées vs présents dans models.yml,

valider hardware_preset / profiles / label_map et existence des DocBin quand on lance train/eval.

E) Tests & samples

Mini-corpus data/sample/... + Make targets :

make sample_prepare PROFILE=ideo_quick

make sample_train PROFILE=ideo_quick FAMILIES=spacy,sklearn,hf

make sample_pipeline PROFILE=ideo_quick

Test de régression équilibrage : script qui compare V2 vs V4 sur un micro-jeu (alpha_total au moins).

F) Logging & reproductibilité

Logger unique (remplacer les print clés par logger.info/warn/error), écriture dans logs/run-*.log.

Seeding étendu : en plus de random, seed numpy / torch si installés, log dans metas.

G) Doc & packaging

Docs : docs/howto_V4.md, docs/ref_V4_parameters.md (finaliser), docs/dev_V4.md (à jour avec arbo + schémas meta).

Packaging light : option pyproject.toml/setup.cfg ou au moins un PYTHONPATH clair dans le Makefile.
