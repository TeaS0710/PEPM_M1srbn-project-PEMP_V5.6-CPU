# Problème de structure du corpus `web1` (idéologie globale & crawls)

## 1. Contexte

Le corpus `web1` est un gros ensemble d’articles web annotés en idéologie globale (`left` / `right`), utilisé pour entraîner et évaluer des modèles de classification (SVM, spaCy, HF, etc.).
Les performances obtenues sont très élevées, parfois proches de scores « parfaits », ce qui a fait suspecter au départ :

- des fuites de données entre train et test,
- ou un sur-apprentissage dû à des doublons massifs.

Les analyses détaillées montrent en réalité que le **problème principal vient de la structure même du corpus** et non d’un bug du pipeline. :contentReference[oaicite:0]{index=0}

---

## 2. Ce que disent les stats

### 2.1. Sites monolabel = tâche presque triviale

- Environ **2 488 sites distincts** (`label_raw`) dans la vue `ideology_global`.
- **100 % des sites sont strictement monolabel** : chaque domaine n’apparaît qu’avec un seul label idéologique (toujours left ou toujours right).
- **100 % des documents** proviennent de ces sites monolabel. :contentReference[oaicite:1]{index=1}

Conséquence directe : une baseline naïve qui prédit simplement *« pour chaque site, son label majoritaire »* atteint déjà :

- **Accuracy ≈ 0,985**
- **Macro-F1 ≈ 0,982**. :contentReference[oaicite:2]{index=2}

Autrement dit, **reconnaître la source suffit presque à prédire l’idéologie**. Les modèles supervisés ne font qu’exploiter cette structure très favorable.

### 2.2. Signaux formels (longueur) très corrélés à l’idéologie

- Les textes `left` sont en moyenne **sensiblement plus longs** que les textes `right`, avec des distributions stables entre train et test. :contentReference[oaicite:3]{index=3}
- Un classifieur utilisant **uniquement la longueur du texte** (logistic regression sur `len(text)`) atteint déjà :

  - **Accuracy ≈ 0,71**
  - **Macro-F1 ≈ 0,61**. :contentReference[oaicite:4]{index=4}

Donc, même sans regarder le contenu lexical, il existe un signal formel non négligeable.

### 2.3. Doublons et quasi-doublons

**Avant nettoyage (corpus brut)** :

- **528 206** documents dans `corpus.xml`. :contentReference[oaicite:5]{index=5}
- Pas de doublons exacts sur `md5_text` (0 empreintes partagées).
- Sur un échantillon TF-IDF, seulement ~**0,7 %** des textes ont un voisin très proche `cos ≥ 0,95`. :contentReference[oaicite:6]{index=6}

**Nettoyages appliqués :**

1. **PATCH V1** → `corpus.cleaned.xml`
   - Déduplication sur `URL` et nettoyage initial.
   - **514 340** documents restants. :contentReference[oaicite:7]{index=7}

2. **PATCH V2** → `corpus.cleaned.V2.xml`
   - Suppression agressive de :
     - **27 905** textes trop courts (`len < 400`),
     - **16 825** doublons `(title, domain)`,
     - **43** quasi-doublons par LSH sur les gros domaines (fdesouche, ER, RevPerm, etc.). :contentReference[oaicite:8]{index=8}
   - Résultat final : **469 567** documents (soit ~8,7 % de docs en moins). :contentReference[oaicite:9]{index=9}

Ces patches réduisent effectivement la redondance, mais les performances des modèles restent **très élevées** après PATCH V2 : le problème ne vient donc pas principalement des doublons, mais de la structure idéologique par source.

---

## 3. Résumé du problème

1. **Structure du corpus :**
   - Les sites sont **quasi parfaitement alignés** avec une seule idéologie.
   - La tâche se rapproche d’une **classification de sources** plutôt que d’une analyse fine du contenu.

2. **Signaux formels et redondance :**
   - La **longueur des textes** est déjà suffisamment corrélée aux labels pour fournir une baseline correcte.
   - Il existe des **quasi-doublons lexicaux**, y compris entre train et test, mais ils ne suffisent pas à expliquer seuls les très bons scores.

3. **Nettoyage :**
   - PATCH V1 et PATCH V2 ont :
     - éliminé les doublons `URL`/`title+domain`,
     - filtré les textes trop courts,
     - supprimé un petit nombre de quasi-doublons haute similarité,
   - tout en conservant un corpus de près de **470k** documents. :contentReference[oaicite:10]{index=10}

4. **Conséquence méthodologique :**
   - Les performances élevées (jusqu’à des scores quasi parfaits) **ne signifient pas que les modèles “comprennent” finement l’idéologie**.
   - Elles reflètent surtout une tâche **structurellement facile**, où connaître le site et quelques propriétés formelles suffit déjà à prédire le label.

En résumé, le **vrai problème** n’est plus un bug du pipeline, ni uniquement des doublons, mais un **biais structurel fort du corpus** :
l’idéologie est pratiquement équivalente au site d’origine, ce qui rend la tâche beaucoup plus simple que ce qu’elle prétend mesurer.
