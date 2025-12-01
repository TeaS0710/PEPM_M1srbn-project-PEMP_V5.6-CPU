```python
python - << 'PY'
from lxml import etree
from collections import Counter
from pathlib import Path
import numpy as np
import re

TEI_PATH = Path("data/raw/web1/corpus.xml")
DOC_TAG  = "TEI"
ns = {"tei": "http://www.tei-c.org/ns/1.0"}

domain_counts = Counter()
class_counts  = Counter()
crawl_counts  = Counter()
lang_counts   = Counter()
md5_text_counts = Counter()
url_counts    = Counter()

lengths = []
n_docs = 0

def get_body_text(doc):
    body = doc.find(".//tei:body", namespaces=ns)
    target = body if body is not None else doc
    text = " ".join(target.itertext())
    return " ".join(text.split())

def get_url(doc):
    url_el = doc.find(".//tei:sourceDesc/tei:biblStruct/tei:idno[@type='url']",
                      namespaces=ns)
    if url_el is not None and url_el.text:
        return url_el.text.strip()
    return None

md5_re   = re.compile(r"md5_text=([0-9a-f]{32})")
chars_re = re.compile(r"chars=(\\d+)")

def get_md5_text(doc):
    change_el = doc.find(".//tei:revisionDesc/tei:change", namespaces=ns)
    if change_el is None or change_el.text is None:
        return None
    m = md5_re.search(change_el.text)
    if m:
        return m.group(1)
    return None

def get_chars(doc, text_len_fallback=None):
    change_el = doc.find(".//tei:revisionDesc/tei:change", namespaces=ns)
    if change_el is not None and change_el.text:
        m = chars_re.search(change_el.text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return text_len_fallback

context = etree.iterparse(str(TEI_PATH), events=("end",), huge_tree=True)
for event, elem in context:
    tag = elem.tag.split('}')[-1]
    if tag != DOC_TAG:
        continue

    n_docs += 1

    text = get_body_text(elem)
    lengths.append(len(text))

    url = get_url(elem)
    if url:
        url_counts[url] += 1

    md5 = get_md5_text(elem)
    if md5:
        md5_text_counts[md5] += 1

    # domain / class / crawl
    for term in elem.findall(".//tei:textClass/tei:keywords/tei:term",
                             namespaces=ns):
        t_type = (term.get("type") or "").strip()
        val    = (term.text or "").strip()
        if not val:
            continue
        if t_type == "domain":
            domain_counts[val] += 1
        elif t_type == "class":
            class_counts[val] += 1
        elif t_type == "crawl":
            crawl_counts[val] += 1

    # langues
    for lang_el in elem.findall(".//tei:langUsage/tei:language",
                                namespaces=ns):
        lang = (lang_el.get("ident") or (lang_el.text or "").strip() or "UNK")
        lang_counts[lang] += 1

    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]

print(f"Nombre total de documents : {n_docs}")

if lengths:
    arr = np.array(lengths)
    print("\\nStats longueur (body, caractères) :")
    print(f"  min   : {arr.min()}")
    print(f"  p01   : {np.percentile(arr, 1):.1f}")
    print(f"  p05   : {np.percentile(arr, 5):.1f}")
    print(f"  méd   : {np.median(arr):.1f}")
    print(f"  moy   : {arr.mean():.1f}")
    print(f"  p95   : {np.percentile(arr, 95):.1f}")
    print(f"  p99   : {np.percentile(arr, 99):.1f}")
    print(f"  max   : {arr.max()}")

print("\\n=== Langues (top 10) ===")
for lang, c in lang_counts.most_common(10):
    print(f"  {lang:10s} {c:8d}")

print("\\n=== Classes (term@type='class', top 10) ===")
for cl, c in class_counts.most_common(10):
    print(f"  {cl:15s} {c:8d}")

print("\\n=== Domaines (term@type='domain', top 20) ===")
for dom, c in domain_counts.most_common(20):
    print(f"  {dom:40s} {c:8d}")

print("\\n=== Crawls (term@type='crawl', top 10) ===")
for cr, c in crawl_counts.most_common(10):
    print(f"  {cr:40s} {c:8d}")

# Stats sur les URL
n_urls = len(url_counts)
n_urls_multi = sum(1 for c in url_counts.values() if c > 1)
n_docs_extra_urls = sum(c - 1 for c in url_counts.values() if c > 1)

print("\\n=== URLs ===")
print(f"  URLs uniques           : {n_urls}")
print(f"  URLs avec >=2 docs     : {n_urls_multi}")
print(f"  Docs 'en trop' (URL)   : {n_docs_extra_urls}")

# Stats sur les md5_text
n_md5 = len(md5_text_counts)
n_md5_multi = sum(1 for c in md5_text_counts.values() if c > 1)
n_docs_extra_md5 = sum(c - 1 for c in md5_text_counts.values() if c > 1)

print("\\n=== md5_text ===")
print(f"  Empreintes uniques       : {n_md5}")
print(f"  Empreintes avec >=2 docs : {n_md5_multi}")
print(f"  Docs 'en trop' (md5)     : {n_docs_extra_md5}")
PY
```

```python
python - << 'PY'
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

TSV_PATH = Path("data/raw/web1/web1_flat.cleaned.tsv")

print(f"[INFO] Chargement de {TSV_PATH} ...")
df = pd.read_csv(TSV_PATH, sep="\t")

df["text_norm"] = df["text_norm"].fillna("").astype(str).str.strip()
df["domain"]    = df["domain"].fillna("").astype(str)
df["doc_id"]    = df["doc_id"].astype(str)

# On ignore les tout petits textes pour l'analyse (souvent bruit / pages vides)
df = df[df["text_norm"].str.len() >= 300].copy()
print(f"[INFO] Après filtrage len>=300 : {len(df)} docs")

# On échantillonne pour pas exploser la RAM
N_MAX = 100_000
if len(df) > N_MAX:
    df_sample = df.sample(n=N_MAX, random_state=42)
    print(f"[INFO] Échantillon aléatoire de {N_MAX} docs")
else:
    df_sample = df.copy()
    print(f"[INFO] Utilisation de tous les docs ({len(df_sample)})")

df_sample = df_sample.reset_index(drop=False).rename(columns={"index": "orig_idx"})
texts = df_sample["text_norm"].tolist()

print("[INFO] Vectorisation TF-IDF ...")
vec = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_features=200_000,
    dtype=np.float32,
)
X = vec.fit_transform(texts)
print(f"[INFO] Matrice TF-IDF : shape={X.shape}")

print("[INFO] Calcul des plus proches voisins (cosine) ...")
nn = NearestNeighbors(
    n_neighbors=6,   # self + 5 voisins
    metric="cosine",
    n_jobs=-1,
)
nn.fit(X)
distances, indices = nn.kneighbors(X, return_distance=True)

# Similarité avec le plus proche voisin distinct (on saute le self à indice 0)
sims = 1.0 - distances[:, 1]
assert sims.shape[0] == len(df_sample)

print("\n=== Distribution du max cosinus (voisin le plus proche) ===")
for q in [0, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  q{q:2d} = {np.percentile(sims, q):.4f}")

print("\n=== Proportion de docs avec un voisin très proche ===")
for thr in [0.8, 0.9, 0.95, 0.98, 0.99]:
    frac = float((sims >= thr).mean())
    print(f"  cos >= {thr:.2f} : {frac*100:6.2f}% des docs")

# Analyse par domaine : quels domaines ont le plus de quasi-doublons ?
print("\n=== Domaines avec le plus forte proportion de voisins cos>=0.95 (>=50 docs) ===")
df_sample["max_cos"] = sims
domain_stats = []
for dom, sub in df_sample.groupby("domain"):
    n = len(sub)
    if n < 50:
        continue
    mask = sub["max_cos"] >= 0.95
    n_high = int(mask.sum())
    frac = n_high / n
    domain_stats.append((dom, n, n_high, frac))

domain_stats.sort(key=lambda x: x[3], reverse=True)
for dom, n, n_high, frac in domain_stats[:20]:
    print(f"  {dom:40s}  n={n:6d}  n_cos>=0.95={n_high:5d} ({frac*100:5.2f}%)")
PY
```

```bash
Nombre total de documents : 528206
\nStats longueur (body, caractères) :
  min   : 5
  p01   : 138.0
  p05   : 380.0
  méd   : 1818.0
  moy   : 4259.8
  p95   : 14738.0
  p99   : 40204.6
  max   : 15279690
\n=== Langues (top 10) ===
  fr           499240
  en            18638
  es             3599
  de             1863
  it             1130
  pt             1020
  ru              757
  ar              569
  ca              267
  ro              113
\n=== Classes (term@type='class', top 10) ===
  CORE              528206
\n=== Domaines (term@type='domain', top 20) ===
  fdesouche.com                              178279
  egaliteetreconciliation.fr                  46719
  revolutionpermanente.fr                     34691
  initiative-communiste.fr                    23853
  actionfrancaise.net                          9579
  polemia.com                                  9450
  contretemps.eu                               8775
  npa-lanticapitaliste.org                     8306
  lafranceinsoumise.fr                         7356
  lundi.am                                     5890
  jean-jaures.org                              5829
  wikipedia.org                                5103
  lefigaro.fr                                  4530
  franceinfo.fr                                3443
  tnova.fr                                     2486
  mediapart.fr                                 2162
  liberation.fr                                1722
  20minutes.fr                                 1565
  nouvelobs.com                                1489
  voltairenet.org                              1348
\n=== Crawls (term@type='crawl', top 10) ===
  crawl-fdesouche-20251003_231247            184710
  crawl-er-20251003_231234                    77676
  crawl-lo-full-20250928_001323               63054
  crawl-revperm-20250928_003219               58538
  crawl-initiative-communiste-20250928_003234    37318
  crawl-contretemps-20250928_003525           19615
  crawl-polemia-20250928_003004               14196
  crawl-lundi-am-20250928_003202              12519
  crawl-ucl-20250928_003510                   12105
  crawl-jean-jaures-20250928_003513           12016
\n=== URLs ===
  URLs uniques           : 421066
  URLs avec >=2 docs     : 1862
  Docs 'en trop' (URL)   : 12066
\n=== md5_text ===
  Empreintes uniques       : 528206
  Empreintes avec >=2 docs : 0
  Docs 'en trop' (md5)     : 0
```
```bash
=== Distribution du max cosinus (voisin le plus proche) ===
  q 0 = 0.0000
  q25 = 0.2570
  q50 = 0.3218
  q75 = 0.4238
  q90 = 0.5837
  q95 = 0.7225
  q99 = 0.9250
  q100 = 1.0000

=== Proportion de docs avec un voisin très proche ===
  cos >= 0.80 :   3.08% des docs
  cos >= 0.90 :   1.31% des docs
  cos >= 0.95 :   0.72% des docs
  cos >= 0.98 :   0.44% des docs
  cos >= 0.99 :   0.29% des docs

=== Domaines avec le plus forte proportion de voisins cos>=0.95 (>=50 docs) ===
  legrandsoir.info                          n=    69  n_cos>=0.95=    9 (13.04%)
  opex360.com                               n=   155  n_cos>=0.95=   20 (12.90%)
  mondialisation.ca                         n=    93  n_cos>=0.95=   11 (11.83%)
  lapresse.ca                               n=    68  n_cos>=0.95=    8 (11.76%)
  voltairenet.org                           n=   269  n_cos>=0.95=   18 ( 6.69%)
  rtbf.be                                   n=    53  n_cos>=0.95=    3 ( 5.66%)
  over-blog.org                             n=    54  n_cos>=0.95=    3 ( 5.56%)
  blogspot.com                              n=   172  n_cos>=0.95=    9 ( 5.23%)
  marianne.net                              n=    52  n_cos>=0.95=    2 ( 3.85%)
  ojim.fr                                   n=    53  n_cos>=0.95=    2 ( 3.77%)
  hypotheses.org                            n=    55  n_cos>=0.95=    2 ( 3.64%)
  lopinion.fr                               n=    57  n_cos>=0.95=    2 ( 3.51%)
  over-blog.com                             n=   116  n_cos>=0.95=    3 ( 2.59%)
  wordpress.com                             n=   254  n_cos>=0.95=    6 ( 2.36%)
  mediapart.fr                              n=   420  n_cos>=0.95=    9 ( 2.14%)
  medias-presse.info                        n=    60  n_cos>=0.95=    1 ( 1.67%)
  ladepeche.fr                              n=    66  n_cos>=0.95=    1 ( 1.52%)
  polemia.com                               n=  1467  n_cos>=0.95=   21 ( 1.43%)
  assemblee-nationale.fr                    n=   144  n_cos>=0.95=    2 ( 1.39%)
  egaliteetreconciliation.fr                n=  9226  n_cos>=0.95=  128 ( 1.39%)
```
