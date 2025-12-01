```bash
cd data/raw/web1
ln -s corpus.cleaned.V2.xml corpus.xml
```
```bash
make clean
```
```bash
make run STAGE=pipeline PROFILE=ideo_quick \
  TRAIN_PROP=0.7 \
  MAX_DOCS_SKLEARN=1000 \
  MAX_DOCS_SPACY=1000 \
  MAX_DOCS_HF=200 \
  OVERRIDES="max_docs_global=1500"
```
```bash
python -m scripts.superior.superior_orchestrator \
  --exp-config configs/superior/exp_ideo_balancing_sweep.yml \
  --parallel 1 \
  --max-ram-gb 8 \
  --max-runs 2 \
  --resume
```

**config superior**
```yml
base:
  profile: ideo_quick
  stage: pipeline
  make_vars:
    HARDWARE_PRESET: small
    TRAIN_PROP: 0.7       # ou 0.8
    MAX_DOCS_SKLEARN: 1000
    MAX_DOCS_SPACY: 1000
    MAX_DOCS_HF: 200
  overrides:
    view: ideology_global
    max_docs_global: 1500
```
