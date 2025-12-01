Original : data/raw/web1/corpus.xml → 528 206 docs
Patch V1 : data/raw/web1/corpus.cleaned.xml → 514 340 docs
Patch V2 : data/raw/web1/corpus.cleaned.V2.xml → 469 567 docs

V2 = V1 – 44 773 docs supprimés (≈ 8,7 %), dont :
    27 905 textes trop courts (len(text) < 400),
    16 825 doublons titre+domain,
    43 quasi-doublons détectés par LSH sur les gros domaines.

```python
python - << 'PY'
from pathlib import Path
import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH

# ------------------------------
# Paramètres PATCH V2 (ajustables)
# ------------------------------
MIN_LEN_SHORT = 400          # drop agressif des textes trop courts
MIN_DOCS_DOMAIN_LSH = 500    # on applique LSH seulement aux domaines avec >= 500 docs
MIN_LEN_LSH = 800            # on ne MinHash que les textes assez longs
LSH_THRESHOLD = 0.9          # seuil Jaccard approximatif (agressif)
NUM_PERM = 64                # nb de permutations MinHash (64 pour limiter RAM)

# ------------------------------
# Fichiers d'entrée/sortie
# ------------------------------
TSV_IN      = Path("data/raw/web1/web1_flat.cleaned.tsv")
TSV_OUT     = Path("data/raw/web1/web1_flat.V2.tsv")
KEEP_PATH   = Path("data/raw/web1/keep_ids.V2.txt")
DROP_PATH   = Path("data/raw/web1/drop_ids.V2.txt")
REPORT_PATH = Path("data/raw/web1/dedup_V2_report.tsv")

print(f"[INFO] Chargement de {TSV_IN} ...")
df = pd.read_csv(TSV_IN, sep="\t")

# Assainir les colonnes nécessaires
for col in ["doc_id", "domain", "url", "md5_text", "title", "text_norm"]:
    if col not in df.columns:
        raise SystemExit(f"Colonne manquante dans {TSV_IN}: {col}")

df["doc_id"]    = df["doc_id"].astype(str)
df["domain"]    = df["domain"].fillna("").astype(str)
df["url"]       = df["url"].fillna("").astype(str)
df["md5_text"]  = df["md5_text"].fillna("").astype(str)
df["title"]     = df["title"].fillna("").astype(str)
df["text_norm"] = df["text_norm"].fillna("").astype(str)

# colonnes de longueur si absentes
if "text_len" not in df.columns:
    df["text_len"] = df["text_norm"].str.len()
if "chars" not in df.columns:
    df["chars"] = df["text_norm"].str.len()

df["text_len"] = df["text_len"].fillna(0).astype(int)
df["chars"]    = df["chars"].fillna(0).astype(int)

# Position globale pour casser les égalités
df["pos"] = np.arange(len(df))

# colonnes de statut
df["kept"] = True
df["drop_reason"] = ""

# mapping doc_id -> index pour maj rapide
id2idx = {doc_id: idx for idx, doc_id in enumerate(df["doc_id"].tolist())}

def add_reason(idx, reason):
    cur = df.at[idx, "drop_reason"]
    if not cur:
        df.at[idx, "drop_reason"] = reason
    else:
        parts = cur.split("+")
        if reason not in parts:
            df.at[idx, "drop_reason"] = cur + "+" + reason

def mark_drop(doc_id, reason):
    idx = id2idx[doc_id]
    if not df.at[idx, "kept"]:
        # déjà drop, on ajoute juste la raison si besoin
        add_reason(idx, reason)
        return False
    df.at[idx, "kept"] = False
    add_reason(idx, reason)
    return True

# ------------------------------
# STEP 1 – Drop agressif des textes trop courts
# ------------------------------
mask_short = (df["text_len"] < MIN_LEN_SHORT) & df["kept"]
short_ids = df.loc[mask_short, "doc_id"].tolist()

dropped_short = 0
for d in short_ids:
    if mark_drop(d, "short"):
        dropped_short += 1

print(f"[STEP1] Docs retirés pour texte trop court (<{MIN_LEN_SHORT}) : {dropped_short}")

# ------------------------------
# STEP 2 – Dédoublonnage titre+domain
# ------------------------------
def normalize_title(t: str) -> str:
    t = t.lower().strip()
    return " ".join(t.split())

df["norm_title"] = df["title"].map(normalize_title)

df_kept = df[df["kept"] & (df["norm_title"].str.len() > 0)].copy()
grouped = df_kept.groupby(["domain", "norm_title"], sort=False)

dropped_title = 0

for (dom, nt), group in grouped:
    if len(group) <= 1:
        continue
    # on garde le plus long, puis le plus "chars", puis le plus ancien
    group_sorted = group.sort_values(
        by=["text_len", "chars", "pos"],
        ascending=[False, False, True]
    )
    keep_doc = group_sorted["doc_id"].iloc[0]
    to_drop = group_sorted["doc_id"].iloc[1:]
    for d in to_drop:
        if mark_drop(d, "title_dup"):
            dropped_title += 1

print(f"[STEP2] Docs retirés par titre+domain dupliqué : {dropped_title}")

# ------------------------------
# STEP 3 – Dédoublonnage LSH (quasi-doublons par domaine)
# ------------------------------
from typing import Iterable

def shingles_3gram(text: str) -> Iterable[str]:
    tokens = text.split()
    for i in range(len(tokens) - 2):
        yield " ".join(tokens[i:i+3])

# on ne regarde que les docs encore kept
df_kept = df[df["kept"]].copy()
domain_counts = df_kept["domain"].value_counts()
domains_big = [dom for dom, c in domain_counts.items() if c >= MIN_DOCS_DOMAIN_LSH and dom]

print(f"[STEP3] Nb de domaines avec >= {MIN_DOCS_DOMAIN_LSH} docs : {len(domains_big)}")

dropped_lsh = 0

for dom in domains_big:
    sub = df_kept[
        (df_kept["domain"] == dom)
        & (df_kept["text_len"] >= MIN_LEN_LSH)
    ].copy()
    n_sub = len(sub)
    if n_sub <= 1:
        continue

    print(f"[STEP3] Domaine '{dom}' : {n_sub} docs (len>={MIN_LEN_LSH}), construction LSH ...")

    # MinHash LSH sur ce domaine
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    minhashes = {}

    for row in sub.itertuples(index=False):
        doc_id = row.doc_id
        text   = row.text_norm
        m = MinHash(num_perm=NUM_PERM)
        for sh in shingles_3gram(text):
            m.update(sh.encode("utf-8"))
        lsh.insert(doc_id, m)
        minhashes[doc_id] = m

    visited = set()
    sub_ids = set(sub["doc_id"].tolist())

    for doc_id, m in minhashes.items():
        if doc_id in visited:
            continue
        near = [d for d in lsh.query(m) if d in sub_ids]
        for d in near:
            visited.add(d)
        if len(near) <= 1:
            continue

        # cluster de quasi-doublons
        # on ne regarde que ceux encore kept au moment du cluster
        cluster_idxs = [id2idx[d] for d in near]
        cluster = df.loc[cluster_idxs]
        cluster = cluster[cluster["kept"]]

        if len(cluster) <= 1:
            continue

        cluster_sorted = cluster.sort_values(
            by=["text_len", "chars", "pos"],
            ascending=[False, False, True]
        )
        keep_doc = cluster_sorted["doc_id"].iloc[0]
        to_drop = cluster_sorted["doc_id"].iloc[1:]

        for d in to_drop:
            if mark_drop(d, "lsh_dup"):
                dropped_lsh += 1

print(f"[STEP3] Docs retirés par LSH (quasi-doublons domaine) : {dropped_lsh}")

# ------------------------------
# Bilan & sauvegarde
# ------------------------------
all_ids = set(df["doc_id"].tolist())
keep_ids = sorted(df[df["kept"]]["doc_id"].tolist())
drop_ids = sorted(list(all_ids - set(keep_ids)))

KEEP_PATH.write_text("\n".join(keep_ids), encoding="utf-8")
DROP_PATH.write_text("\n".join(drop_ids), encoding="utf-8")

df.to_csv(REPORT_PATH, sep="\t", index=False)
df[df["kept"]].to_csv(TSV_OUT, sep="\t", index=False)

print("\n[PATCH V2] BILAN")
print(f"  Docs totaux     : {len(df)}")
print(f"  Docs gardés     : {len(keep_ids)}")
print(f"  Docs supprimés  : {len(drop_ids)}")
print(f"    - short       : {dropped_short}")
print(f"    - title_dup   : {dropped_title}")
print(f"    - lsh_dup     : {dropped_lsh}")
print(f"[OUT] keep_ids    -> {KEEP_PATH}")
print(f"[OUT] drop_ids    -> {DROP_PATH}")
print(f"[OUT] TSV V2      -> {TSV_OUT}")
print(f"[OUT] report V2   -> {REPORT_PATH}")
PY
```

```python
python - << 'PY'
from lxml import etree
from pathlib import Path

TEI_IN   = Path("data/raw/web1/corpus.cleaned.xml")      # PATCH V1
TEI_OUT  = Path("data/raw/web1/corpus.cleaned.V2.xml")   # PATCH V2
KEEP_IDS = Path("data/raw/web1/keep_ids.V2.txt")

DOC_TAG = "TEI"

keep_ids = set(
    line.strip()
    for line in KEEP_IDS.read_text(encoding="utf-8").splitlines()
    if line.strip()
)
print(f"[INFO] Nb doc_id gardés (PATCH V2) : {len(keep_ids)}")

def get_doc_id(doc, fallback_idx):
    doc_id = doc.get("{http://www.w3.org/XML/1998/namespace}id")
    if doc_id:
        return doc_id
    doc_id = doc.get("id")
    if doc_id:
        return doc_id
    return f"doc_{fallback_idx:06d}"

with TEI_OUT.open("wb") as out:
    out.write(b"<?xml version='1.0' encoding='utf-8'?>\n")
    out.write(b"<teiCorpus xmlns=\"http://www.tei-c.org/ns/1.0\">\n")

    context = etree.iterparse(str(TEI_IN), events=("end",), huge_tree=True)
    idx = 0
    kept = 0
    skipped = 0

    for event, elem in context:
        tag = elem.tag.split('}')[-1]
        if tag != DOC_TAG:
            continue

        idx += 1
        doc_id = get_doc_id(elem, idx)
        if doc_id in keep_ids:
            out.write(etree.tostring(elem, encoding="utf-8"))
            out.write(b"\n")
            kept += 1
        else:
            skipped += 1

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    out.write(b"</teiCorpus>\n")

print(f"[DONE] Docs écrits (PATCH V2) : {kept}")
print(f"[DONE] Docs ignorés           : {skipped}")
print(f"[OK] Corpus PATCH V2 écrit dans : {TEI_OUT}")
PY
```

```bash
# Position globale pour casser les égalités
PYint(f"[OUT] report V2   -> {REPORT_PATH}"))x=False)tf-8")e) : {dropped_lsh}")u
[INFO] Chargement de data/raw/web1/web1_flat.cleaned.tsv ...
[STEP1] Docs retirés pour texte trop court (<400) : 27905
[STEP2] Docs retirés par titre+domain dupliqué : 16825
[STEP3] Nb de domaines avec >= 500 docs : 37
[STEP3] Domaine 'fdesouche.com' : 143637 docs (len>=800), construction LSH ...
[STEP3] Domaine 'egaliteetreconciliation.fr' : 39931 docs (len>=800), construction LSH ...
[STEP3] Domaine 'revolutionpermanente.fr' : 33467 docs (len>=800), construction LSH ...
[STEP3] Domaine 'initiative-communiste.fr' : 13975 docs (len>=800), construction LSH ...
[STEP3] Domaine 'polemia.com' : 6502 docs (len>=800), construction LSH ...
[STEP3] Domaine 'lafranceinsoumise.fr' : 4922 docs (len>=800), construction LSH ...
[STEP3] Domaine 'actionfrancaise.net' : 4114 docs (len>=800), construction LSH ...
[STEP3] Domaine 'contretemps.eu' : 5190 docs (len>=800), construction LSH ...
[STEP3] Domaine 'lundi.am' : 5165 docs (len>=800), construction LSH ...
[STEP3] Domaine 'jean-jaures.org' : 4193 docs (len>=800), construction LSH ...
[STEP3] Domaine 'wikipedia.org' : 5004 docs (len>=800), construction LSH ...
[STEP3] Domaine 'lefigaro.fr' : 4271 docs (len>=800), construction LSH ...
[STEP3] Domaine 'npa-lanticapitaliste.org' : 3705 docs (len>=800), construction LSH ...
[STEP3] Domaine 'franceinfo.fr' : 3334 docs (len>=800), construction LSH ...
[STEP3] Domaine 'mediapart.fr' : 1729 docs (len>=800), construction LSH ...
[STEP3] Domaine 'liberation.fr' : 1687 docs (len>=800), construction LSH ...
[STEP3] Domaine '20minutes.fr' : 1467 docs (len>=800), construction LSH ...
[STEP3] Domaine 'nouvelobs.com' : 1398 docs (len>=800), construction LSH ...
[STEP3] Domaine 'tnova.fr' : 1231 docs (len>=800), construction LSH ...
[STEP3] Domaine 'voltairenet.org' : 1303 docs (len>=800), construction LSH ...
[STEP3] Domaine 'lepoint.fr' : 1329 docs (len>=800), construction LSH ...
[STEP3] Domaine 'wordpress.com' : 1161 docs (len>=800), construction LSH ...
[STEP3] Domaine 'bfmtv.com' : 1061 docs (len>=800), construction LSH ...
[STEP3] Domaine 'blogspot.com' : 894 docs (len>=800), construction LSH ...
[STEP3] Domaine 'laizquierdadiario.com' : 818 docs (len>=800), construction LSH ...
[STEP3] Domaine 'opex360.com' : 757 docs (len>=800), construction LSH ...
[STEP3] Domaine 'radiofrance.fr' : 721 docs (len>=800), construction LSH ...
[STEP3] Domaine 'lexpress.fr' : 678 docs (len>=800), construction LSH ...
[STEP3] Domaine 'parti-socialiste.fr' : 626 docs (len>=800), construction LSH ...
[STEP3] Domaine 'theguardian.com' : 663 docs (len>=800), construction LSH ...
[STEP3] Domaine 'over-blog.com' : 611 docs (len>=800), construction LSH ...
[STEP3] Domaine '20min.ch' : 557 docs (len>=800), construction LSH ...
[STEP3] Domaine 'hizb-ut-tahrir.info' : 506 docs (len>=800), construction LSH ...
[STEP3] Domaine 'assemblee-nationale.fr' : 408 docs (len>=800), construction LSH ...
[STEP3] Domaine 'huffingtonpost.fr' : 533 docs (len>=800), construction LSH ...
[STEP3] Domaine 'republicains.fr' : 465 docs (len>=800), construction LSH ...
[STEP3] Domaine 'francebleu.fr' : 501 docs (len>=800), construction LSH ...
[STEP3] Docs retirés par LSH (quasi-doublons domaine) : 43

[PATCH V2] BILAN
  Docs totaux     : 514340
  Docs gardés     : 469567
  Docs supprimés  : 44773
    - short       : 27905
    - title_dup   : 16825
    - lsh_dup     : 43
[OUT] keep_ids    -> data/raw/web1/keep_ids.V2.txt
[OUT] drop_ids    -> data/raw/web1/drop_ids.V2.txt
[OUT] TSV V2      -> data/raw/web1/web1_flat.V2.tsv
[OUT] report V2   -> data/raw/web1/dedup_V2_report.tsv
```
```bash
[INFO] Nb doc_id gardés (PATCH V2) : 469567
[DONE] Docs écrits (PATCH V2) : 469567
[DONE] Docs ignorés           : 44773
[OK] Corpus PATCH V2 écrit dans : data/raw/web1/corpus.cleaned.V2.xml
```
