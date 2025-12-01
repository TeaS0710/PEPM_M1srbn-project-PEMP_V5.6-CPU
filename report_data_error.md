## 1. Tableau récapitulatif des statistiques

### Synthèse globale des analyses (corpus `web1 / ideology_global`)

| Catégorie                                          | Mesure                                       | Valeur                                                                                                  | Commentaire                                                             |
| -------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Corpus & splits**                                | Taille `train.tsv`                           | **50 000 docs**                                                                                         | Ensemble d’entraînement complet (avant split interne 80/20 éventuel).   |
|                                                    | Taille `job.tsv`                             | **50 599 docs**                                                                                         | Ensemble d’évaluation externe.                                          |
|                                                    | Taille totale                                | **100 599 docs**                                                                                        | `train` + `job`.                                                        |
| **Répartition des labels (train)**                 | Count left                                   | **20 371** (40,7 %)                                                                                     | Classe minoritaire.                                                     |
|                                                    | Count right                                  | **29 629** (59,3 %)                                                                                     | Classe majoritaire.                                                     |
|                                                    | Baseline “toujours right” (train)            | **0,593**                                                                                               | Accuracy en prédisant toujours la classe majoritaire sur train.         |
| **Répartition des labels (job)**                   | Count left                                   | **16 241** (32,1 %)                                                                                     | Minoritaire dans job.                                                   |
|                                                    | Count right                                  | **34 358** (67,9 %)                                                                                     | Majoritaire dans job.                                                   |
|                                                    | Baseline “toujours right” (job)              | **0,679**                                                                                               | Accuracy en prédisant toujours right.                                   |
|                                                    | Différence de proportion job–train           | right : **+0,086** ; left : **−0,086**                                                                  | Job est plus “à droite” que train.                                      |
| **Structure par sites (`label_raw`)**              | Nb de sites distincts                        | **2 488**                                                                                               | Calculé sur train+job.                                                  |
|                                                    | Sites strictement monolabel (max_prop = 1.0) | **2 488 / 2 488**                                                                                       | **100 %** des sites n’ont qu’un seul label (left ou right).             |
|                                                    | Sites quasi-monolabel (≥ 0,95)               | **2 488 / 2 488**                                                                                       | Identique, tous strictement monolabel.                                  |
|                                                    | Proportion de docs issus de sites monolabel  | **1,0** (100 %)                                                                                         | Tous les documents proviennent de sources 100 % gauche ou 100 % droite. |
| **Longueur des textes – global (train)**           | Moyenne (chars)                              | **1431,65**                                                                                             | Longueur moyenne.                                                       |
|                                                    | Médiane                                      | **1366**                                                                                                | Valeur centrale.                                                        |
|                                                    | Min / Max                                    | **200 / 30 015**                                                                                        | Textes assez longs, outliers très longs.                                |
|                                                    | 1 % / 5 % / 95 % / 99 %                      | 275 / 411 / 2780 / 3115                                                                                 | Distribution relativement étalée.                                       |
|                                                    | Prop. `< 200` chars                          | **0,0**                                                                                                 | Aucun texte très court (seuil 200).                                     |
|                                                    | Prop. `> 5 000` chars                        | ≈ **0,0**                                                                                               | Très peu de très longs (mais max à 30k).                                |
| **Longueur des textes – par label (train)**        | Left – count                                 | **20 371**                                                                                              | Idem count des labels.                                                  |
|                                                    | Left – mean / std                            | **1612,68 / 843,29**                                                                                    | Textes plus longs et plus dispersés.                                    |
|                                                    | Left – min / Q1 / médiane / Q3 / max         | 200 / 930,5 / 1637 / 2232 / 30 015                                                                      | Distribution plutôt haute.                                              |
|                                                    | Right – count                                | **29 629**                                                                                              | Idem.                                                                   |
|                                                    | Right – mean / std                           | **1307,18 / 577,69**                                                                                    | Textes plus courts en moyenne.                                          |
|                                                    | Right – min / Q1 / médiane / Q3 / max        | 207 / 883 / 1274 / 1612 / 9 043                                                                         | Distribution plus compacte.                                             |
| **Longueur des textes – global (job)**             | Moyenne (chars)                              | **1405,19**                                                                                             | Très proche de train.                                                   |
|                                                    | Médiane                                      | **1343**                                                                                                | -                                                                       |
|                                                    | Min / Max                                    | **200 / 9 850**                                                                                         | Pas d’outliers extrêmes comme 30k mais max élevé.                       |
|                                                    | 1 % / 5 % / 95 % / 99 %                      | 280 / 421 / 2738 / 3119                                                                                 | Distribution similaire à train.                                         |
| **Longueur des textes – par label (job)**          | Left – count                                 | **16 241**                                                                                              | -                                                                       |
|                                                    | Left – mean / std                            | **1614,47 / 827,48**                                                                                    | Très proche du left train.                                              |
|                                                    | Left – min / Q1 / médiane / Q3 / max         | 200 / 919 / 1643 / 2239 / 7 253                                                                         | -                                                                       |
|                                                    | Right – count                                | **34 358**                                                                                              | -                                                                       |
|                                                    | Right – mean / std                           | **1306,26 / 579,27**                                                                                    | Très proche du right train.                                             |
|                                                    | Right – min / Q1 / médiane / Q3 / max        | 204 / 882,25 / 1270 / 1605 / 9 850                                                                      | -                                                                       |
| **Baselines de classification (job)**              | Baseline “toujours right”                    | Accuracy **0,679** ; Macro-F1 **0,404**                                                                 | Exploite uniquement le déséquilibre global.                             |
|                                                    | Baseline “site-majoritaire”                  | Accuracy **0,985** ; Macro-F1 **0,982**                                                                 | Utilise seulement `label_raw → label` (site → idéologie).               |
|                                                    | Site-majoritaire – classe left               | P **1,000** ; R **0,952** ; F1 **0,975** (16 241 docs)                                                  | Très peu d’erreurs sur left.                                            |
|                                                    | Site-majoritaire – classe right              | P **0,978** ; R **1,000** ; F1 **0,989** (34 358 docs)                                                  | Quasi parfait sur right.                                                |
|                                                    | Baseline “longueur seule” (logreg)           | AUC **0,614** ; Accuracy **0,709** ; Macro-F1 **0,610**                                                 | Uniquement `len(text)` comme feature.                                   |
|                                                    | Longueur seule – classe 0 (left)             | P **0,587** ; R **0,319** ; F1 **0,414** (16 241)                                                       | Faible rappel sur left.                                                 |
|                                                    | Longueur seule – classe 1 (right)            | P **0,735** ; R **0,894** ; F1 **0,807** (34 358)                                                       | Très bonne détection de right via la longueur.                          |
| **Doublons exacts**                                | Nb de textes uniques                         | **100 599**                                                                                             | Autant que de documents.                                                |
|                                                    | Nb de docs impliqués dans un doublon exact   | **0**                                                                                                   | Aucun texte strictement dupliqué.                                       |
|                                                    | Proportion de docs dupliqués                 | **0,0 %**                                                                                               | Pas de copie stricte train/job.                                         |
| **Quasi-doublons (TF-IDF, sim ≥ 0,95)**            | Matrice TF-IDF                               | shape **(100 599, 499 861)** ; nnz **28 713 336**                                                       | Unigrams + bigrams, `min_df=5`, float32.                                |
|                                                    | Nb total de paires quasi-doublons            | **49 573**                                                                                              | Paires (doc i, doc j) avec sim ≥ 0,95.                                  |
|                                                    | Nb de docs impliqués                         | **1 115**                                                                                               | ≈ **1,1 %** des documents.                                              |
|                                                    | Paires même label                            | **48 950** (98,7 %)                                                                                     | Redondance majoritairement cohérente.                                   |
|                                                    | Paires label différent                       | **623** (1,3 %)                                                                                         | Cas de bruit / ambiguïté.                                               |
|                                                    | Paires train–train                           | **14 251** (~28,7 %)                                                                                    | Quasi-doublons internes à train.                                        |
|                                                    | Paires job–job                               | **10 582** (~21,3 %)                                                                                    | Quasi-doublons internes à job.                                          |
|                                                    | Paires cross train–job                       | **24 740** (~49,9 %)                                                                                    | Quasi-doublons inter-splits (train↔job).                                |
|                                                    | Paires même site (`label_raw`)               | **27 227** (~54,9 %)                                                                                    | Reprises / variantes au sein d’un même site.                            |
|                                                    | Paires sites différents                      | **22 346** (~45,1 %)                                                                                    | Copie/reprise entre sites différents (souvent alignés).                 |
| **Modèles supervisés (résumé sur tests externes)** | Taille d’un job test (ex.)                   | 450 docs (404 left / 46 right)                                                                          | Selon les rapports `classification_report`.                             |
|                                                    | Gamme d’accuracy                             | **0,90 – 0,95**                                                                                         | Selon le modèle (SVM, spaCy, etc.).                                     |
|                                                    | F1(left)                                     | **0,94 – 0,97**                                                                                         | Très bon sur la classe majoritaire.                                     |
|                                                    | F1(right)                                    | **0,42 – 0,73**                                                                                         | Plus variable, souvent beaucoup plus faible.                            |
|                                                    | Macro-F1                                     | **~0,67 – 0,85**                                                                                        | Loin de 1 : la minoritaire reste difficile.                             |
|                                                    | Test 300 docs (269 left / 31 right)          | Accuracy **0,81 – 0,90**, F1(left) **0,89 – 0,94**, F1(right) **0,37 – 0,39**, Macro-F1 **0,64 – 0,66** | Confirme le même pattern : très bon sur left, F1 modeste sur right.     |

---

## 2. Présentation technique des résultats

### 2.1. Nature du corpus

Le corpus `web1 / ideology_global` contient environ **100k articles web** annotés en deux classes idéologiques (`left` / `right`). La répartition des labels est **modérément déséquilibrée**, avec une surreprésentation de `right` :

* train : 59,3 % `right` / 40,7 % `left`,
* job  : 67,9 % `right` / 32,1 % `left`.

La **baseline “toujours right”** atteint déjà **67,9 %** d’accuracy sur le job, ce qui impose de considérer des métriques balancées (macro-F1, F1 par classe) pour évaluer les modèles.

Le point clef est la **structure par sources (`label_raw`)** :

* 2 488 sites distincts,
* **100 %** de ces sites sont **strictement monolabel**,
* et **100 % des documents** proviennent de ces sites monolabel.

Autrement dit, dans ce corpus, l’idéologie globale est essentiellement *portée par la source* : un site donné est soit toujours classé gauche, soit toujours classé droite.

### 2.2. Propriétés formelles : longueur des textes

L’analyse de la longueur des textes montre que :

* Les articles `left` sont **en moyenne ~300 caractères plus longs** que les `right`,
  et ce, **dans les deux splits** (train et job).
* La distribution de longueur est stable entre split et entre classes.

Cet écart de longueur n’est pas négligeable et introduit un **signal purement formel** : un modèle peut déjà inférer une partie de l’idéologie sans regarder le contenu sémantique, uniquement via la longueur.

La baseline “longueur seule” (régression logistique sur `len(text)`) confirme ce point :

* **AUC = 0,614**,
* **Accuracy = 0,709** (meilleure que la baseline “toujours right”),
* **Macro-F1 = 0,610**.

En particulier, pour la classe `right`, on obtient déjà **F1 ≈ 0,807**, essentiellement grâce à la longueur.

### 2.3. Baselines structurées : la puissance de `label_raw`

Une baseline “site-majoritaire” extrêmement simple est construite comme suit :

1. Sur `train`, on associe à chaque site (`label_raw`) son label majoritaire (en pratique unique).
2. Sur `job`, on prédit pour chaque document le label majoritaire de son site (ou, pour les sites inconnus, le label globalement majoritaire).

Cette baseline atteint :

* **Accuracy = 0,985**,
* **Macro-F1 = 0,982**.

Les F1 par classe sont très élevées :

* left : **F1 = 0,975**,
* right : **F1 = 0,989**.

On dépasse donc nettement les performances des modèles supervisés “classiques” (SVM, spaCy), avec un système qui ne fait qu’exploiter la structure **site → idéologie**.

Cela montre que, tel qu’il est construit, le corpus définit une tâche de classification **très fortement couplée aux sources** : reconnaître le site d’origine permet presque à lui seul de prédire l’idéologie.

### 2.4. Redondance lexicale : quasi-doublons inter-splits

Aucun **doublon exact** (texte identique) n’a été détecté : les 100 599 documents sont tous distincts sur la chaîne brute.

En revanche, une analyse de **quasi-doublons** à l’aide de TF-IDF (unigrams+bigrammes, `min_df=5`) et de la similarité cosinus (seuil 0,95) révèle :

* **49 573 paires** de textes très similaires,
* impliquant **1 115 documents**, soit environ **1,1 %** du corpus.

Cette redondance est très largement **cohérente idéologiquement** :

* **98,7 %** des paires ont **le même label**,
* **1,3 %** seulement sont en désaccord, ce qui constitue un indicateur de **bruit d’annotation** ou de cas ambigus.

La répartition par split montre que :

* environ **50 %** des paires sont **cross train–job**,
* le reste se répartit en paires **train–train** (~28,7 %) et **job–job** (~21,3 %).

Ainsi, un certain nombre de documents du job sont quasi identiques à des documents du train, ce qui peut faciliter la tâche des modèles (ils retrouvent des patterns presque déjà vus).

Enfin, environ **55 %** des paires concernent des documents du **même site**, et 45 % des **sites différents**, ce qui est cohérent avec des phénomènes de reprises/copie entre sources idéologiquement alignées.

### 2.5. Résultats des modèles supervisés

Les différents modèles évalués (SVM, variantes SMO, TextCat spaCy, etc.) donnent des résultats typiques du type :

* sur un job de 450 documents (404 left / 46 right) :

  * accuracy ≈ **0,90–0,95**,
  * F1(left) ≈ **0,94–0,97**,
  * F1(right) ≈ **0,42–0,73**,
  * macro-F1 ≈ **0,67–0,85** ;

* sur un job de 300 documents (269 left / 31 right) :

  * accuracy ≈ **0,81–0,90**,
  * F1(left) ≈ **0,89–0,94**,
  * F1(right) ≈ **0,37–0,39**,
  * macro-F1 ≈ **0,64–0,66**.

Les modèles sont donc :

* **très performants sur la classe majoritaire** (F1 proche de 0,95–0,97),
* **nettement moins bons sur la classe minoritaire**, avec une F1 parfois autour de 0,4–0,7,
* et les macro-F1 restent **loin de 1**.

La proximité des performances entre **modèles linéaires (SVM, logistic)** et **modèles neuronaux spaCy** s’explique par le fait qu’ils exploitent tous les deux un espace de features très similarisé (bag-of-words / n-grams) sur une tâche **structurellement facile** (séparation linéaire bien marquée, sites monolabel).

---

## 3. Conclusion

L’ensemble des analyses converge vers la même interprétation :

1. **Le corpus est fortement structuré par les sources** :

   * Tous les sites (`label_raw`) sont strictement monolabel,
   * Une baseline “site-majoritaire” atteint **0,985 d’accuracy** et **0,982 de macro-F1**,
     → la tâche est essentiellement : *reconnaître des sources idéologiquement alignées*.

2. **Des signaux formels (longueur) sont corrélés à l’idéologie** :

   * Les textes de gauche sont significativement plus longs que ceux de droite,
   * Un classifieur basé sur la seule longueur atteint **0,709 d’accuracy** et **0,610 de macro-F1**,
     → l’idéologie telle que définie dans le corpus est partiellement prédictible par des caractéristiques superficielles.

3. **Une redondance non négligeable existe entre train et job** :

   * ≈ 1,1 % des documents font partie de quasi-doublons lexicaux (sim TF-IDF ≥ 0,95),
   * ≈ 50 % de ces paires relient train et job,
     → certains contenus du job sont très proches de contenus déjà vus en train.

4. **Les “bons résultats” des modèles reflètent avant tout la structure des données** :

   * Les accuracies élevées (0,90–0,95) et les F1 majoritaires proches de 0,95 s’expliquent naturellement par :

     * le couplage fort **site → idéologie**,
     * des signaux formels marqués,
     * et une certaine redondance inter-splits.
   * Dans ce contexte, les modèles linéaires (SVM) peuvent atteindre des performances comparables aux modèles neuronaux spaCy, car l’espace TF-IDF est déjà fortement séparé.

En résumé :

> Les performances élevées observées ne sont pas le signe d’un “sur-apprentissage magique” ou d’un bug, mais la conséquence directe de la manière dont le corpus a été construit (annotation par sources monolabel, forte redondance, signaux formels corrélés aux labels). La tâche évaluée mesure principalement la capacité des modèles à reconnaître des sources alignées idéologiquement dans un espace lexical très structuré, plus qu’une compréhension fine des nuances idéologiques au niveau du texte individuel.
