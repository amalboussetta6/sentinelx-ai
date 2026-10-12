# SentinelX AI — Maintenance prédictive & Maintenance Copilot RAG

Plateforme logicielle de maintenance prédictive : à partir des séries temporelles de capteurs du dataset NASA C-MAPSS, SentinelX AI vise à estimer la **durée de vie utile restante (RUL)** de moteurs, détecter les comportements anormaux, calculer un **Health Score** interprétable et assister la maintenance avec un **copilote RAG** aux réponses sourcées.

> Projet d'intégration en Intelligence Artificielle — prototype académique d'aide à la décision, **100 % logiciel** : les données publiques sont rejouées comme un flux temps réel, sans capteur physique.

**Dernière mise à jour : 12 octobre 2026 — étape terminée : S2 (données + EDA). Prochaine étape : S3 (baseline RUL).**

---

## Sommaire

1. [Présentation du projet](#1-présentation-du-projet)
2. [Périmètre et versions du produit](#2-périmètre-et-versions-du-produit)
3. [Architecture cible et stack](#3-architecture-cible-et-stack)
4. [État d'avancement](#4-état-davancement)
5. [Structure du dépôt](#5-structure-du-dépôt)
6. [Installation](#6-installation)
7. [Données](#7-données)
8. [Analyse exploratoire (EDA) — travail réalisé](#8-analyse-exploratoire-eda--travail-réalisé)
9. [Règles méthodologiques non négociables](#9-règles-méthodologiques-non-négociables)
10. [Prochaine étape détaillée : 02_baseline](#10-prochaine-étape-détaillée--02_baseline)
11. [Feuille de route complète (S4 → S12)](#11-feuille-de-route-complète-s4--s12)
12. [Reprendre le projet dans une nouvelle conversation](#12-reprendre-le-projet-dans-une-nouvelle-conversation)
13. [Environnement technique et pièges connus](#13-environnement-technique-et-pièges-connus)
14. [Limites actuelles](#14-limites-actuelles)
15. [Références](#15-références)

---

## 1. Présentation du projet

### Problématique

Une panne non anticipée immobilise un équipement et perturbe la production. Les données historiques de capteurs permettent d'estimer la dégradation, mais elles sont difficiles à interpréter rapidement. SentinelX AI transforme ces séries temporelles en indicateurs opérationnels : anomalie, état de santé, durée de vie restante et justification exploitable.

> *Comment concevoir une plateforme d'IA capable d'anticiper la dégradation d'équipements à partir de séries temporelles industrielles, de simuler leur suivi en temps réel et d'assister la maintenance grâce à un copilote RAG fiable et sourcé ?*

### Objectifs spécifiques

- Construire une pipeline reproductible de préparation des séries temporelles.
- Établir un **baseline classique** avant le modèle deep learning.
- Prédire le **RUL** sur le dataset principal (C-MAPSS FD001 d'abord).
- Détecter des états anormaux et calculer un **Health Score** interprétable.
- Exposer les modèles via une **API versionnée** (FastAPI).
- Simuler l'arrivée progressive des données sans matériel physique.
- Visualiser flotte, tendances, alertes et incertitude dans un **dashboard** (React).
- Interroger une base documentaire de maintenance avec **RAG + citations**.
- Mesurer séparément performance ML, qualité RAG, latence et robustesse.
- Conteneuriser la solution (Docker Compose) et produire un dépôt reproductible.

### Questions de recherche

1. Un modèle séquentiel (LSTM/GRU) apporte-t-il un gain significatif par rapport à un baseline tabulaire ?
2. Quelle longueur de fenêtre temporelle offre le meilleur compromis performance / latence ?
3. Quel seuil d'alerte maximise le rappel des situations critiques sans trop de faux positifs ?
4. Le reranking / la recherche hybride améliore-t-il réellement la pertinence du RAG ?
5. Le système sait-il s'abstenir lorsque la documentation ne contient pas la réponse ?

---

## 2. Périmètre et versions du produit

| Version | Contenu | Statut attendu |
|---|---|---|
| **MVP** | C-MAPSS + EDA + baseline RUL + FastAPI + simulateur + dashboard de base | Obligatoire |
| **V1** | LSTM/GRU + anomaly score + Health Score + alertes + PostgreSQL + Docker Compose | Obligatoire |
| **V1.5** | RAG sourcé + authentification simple + MLflow + tests + CI | Fortement recommandé |
| **V2** | Digital Twin data-driven + XAI + agent outillé + hybrid retrieval / reranking | Optionnel si le cœur est stable |
| Extension | Détection d'anomalies audio sur MIMII, module séparé | Bonus |

**Hors périmètre :** commande d'une machine réelle, diagnostic certifié, prédiction de la cause physique exacte d'une panne sans labels adaptés, fusion de datasets non synchronisés présentée comme un modèle multimodal, Kubernetes avant stabilisation locale.

**Règle scientifique :** une information qui n'est pas réellement prédite par le modèle ne doit jamais être présentée comme un diagnostic IA.

### Exigences fonctionnelles (résumé)

| ID | Fonction | Priorité | État |
|---|---|---|---|
| F-01 | Ingestion du dataset C-MAPSS | Must | Fait (chargement dans l'EDA) |
| F-02 | EDA : distributions, valeurs manquantes, variabilité, corrélations, trajectoires | Must | **Fait** |
| F-03 | Feature pipeline : scaling, sélection, fenêtrage, cible RUL | Must | Préparé (décisions exportées) |
| F-04 | Au moins un baseline classique reproductible | Must | À faire (S3) |
| F-05 | Au moins un modèle séquentiel (LSTM/GRU) | Must | À faire (S4) |
| F-06 | Comparaison train / validation / test avec artefacts | Must | À faire |
| F-07 | Simulateur : rejouer les cycles d'un moteur à vitesse configurable | Must | À faire (S6) |
| F-08 | Live API : WebSocket/SSE vers le frontend | Must | À faire (S6) |
| F-09 | Health Score 0-100 documenté | Must | À faire (S5) |
| F-10 | Alertes Warning/Critical horodatées et justifiées | Must | À faire (S5) |
| F-11 / F-12 | Ingestion RAG avec métadonnées ; réponses sourcées avec abstention | Should | À faire (S8-S9) |
| F-13 / F-14 | Digital twin ; explicabilité (XAI) | Should | À faire (S10) |
| F-15 / F-16 | Authentification Admin/Viewer ; export de rapport | Could | Optionnel |

Exigences non fonctionnelles principales : reproductibilité via README + Docker Compose, mise à jour live < 2 s, API documentée par OpenAPI, traçabilité de chaque prédiction (modèle, version, scénario, cycle), aucun secret dans Git, réponse RAG sans citation jugée non acceptable.

---

## 3. Architecture cible et stack

| Composant | Responsabilité | Technologie |
|---|---|---|
| data-pipeline | Ingestion, EDA, preprocessing, fenêtres, splits | Python, pandas, NumPy, scikit-learn |
| training | Baselines, deep learning, évaluation | scikit-learn, PyTorch |
| simulator | Replay des cycles et gestion des scénarios | Python async |
| backend | REST, WebSocket, orchestration, authentification | FastAPI |
| database | Machines, sessions, prédictions, alertes, utilisateurs | PostgreSQL |
| vector-db | Chunks documentaires et embeddings | Qdrant |
| rag-service | Retrieval, reranking, prompting, citations | Python + embeddings / LLM |
| frontend | Dashboard, graphiques, alertes, copilote | React + Vite |
| digital-twin | Vue data-driven 2D/3D | React + Three.js |
| mlops | Suivi d'expériences et artefacts | MLflow |
| deployment | Lancement multi-services | Docker Compose |

**Flux d'une session de démonstration :** l'utilisateur choisit un moteur → le simulateur publie le cycle suivant → le backend prépare la fenêtre → le modèle calcule RUL + anomalie → un moteur de règles produit Health Score et niveau d'alerte → la prédiction est persistée et poussée au dashboard → le copilote peut utiliser cet état comme contexte et rechercher les procédures pertinentes.

**Health Score (exemple à documenter, pas une vérité scientifique) :** `Health = 100 × [0,6 × RUL_norm + 0,4 × (1 − anomaly_norm)]`, avec Healthy ≥ 70, Warning 40-69, Critical < 40. Poids et seuils à régler sur la validation, avec analyse des faux positifs / faux négatifs.

---

## 4. État d'avancement

### Planning sur 12 semaines

| Semaine | Focus | Definition of Done | État |
|---|---|---|---|
| S1 | Cadrage + dépôt | README + squelette + issues | Partiel (dépôt et environnement en place) |
| **S2** | **Données + EDA** | **Notebook EDA + échantillon traité** | **Terminé** |
| S3 | Baseline RUL | `baseline_metrics.json` + graphiques | **Prochaine étape** |
| S4 | LSTM / GRU | `model.pt` + rapport comparatif | À faire |
| S5 | Anomalies + Health Score | Module health + tests | À faire |
| S6 | Backend + simulateur | API utilisable via Swagger | À faire |
| S7 | Dashboard MVP | MVP démontrable | À faire |
| S8 | Ingestion RAG | Indexation + benchmark de retrieval | À faire |
| S9 | Copilote | Copilote intégré (citations, abstention) | À faire |
| S10 | Digital Twin + XAI | Fonctionnalité « wow » prête | À faire |
| S11 | Docker + MLflow + tests | Release candidate | À faire |
| S12 | Rapport + soutenance | v1.0 + rapport + slides | À faire |

**Règle de retard :** si plus d'une semaine de retard à S7, supprimer d'abord la 3D, puis MLflow avancé, puis l'agent outillé. Ne jamais sacrifier l'évaluation des modèles, la démo end-to-end ni les citations RAG.

### Backlog (extrait)

| ID | Tâche | Priorité | État |
|---|---|---|---|
| B01 | Configurer dépôt + environnement | Must | En cours (ce dépôt) |
| B02 | Télécharger FD001 | Must | Fait |
| B03 | EDA + dictionnaire des données | Must | Fait |
| B04 | Split par moteur | Must | Fait (figé et exporté) |
| B05 | Labels RUL | Must | Fait dans l'EDA ; à déplacer dans un module (`ml/src/`) |
| B06 | Modèle baseline | Must | À faire |
| B07 | Dataset fenêtré | Must | À faire |
| B08-B09 | LSTM/GRU + rapport d'évaluation | Must | À faire |
| B10-B11 | Score d'anomalie + Health Score | Must | À faire |
| B12-B17 | API, simulateur, PostgreSQL, WebSocket, dashboard | Must | À faire |
| B18-B24 | Corpus RAG, Qdrant, citations, abstention, digital twin, XAI | Should | À faire |
| B25-B30 | MLflow, Docker Compose, tests, démo, README portfolio, vidéo | Must / Should | À faire |

---

## 5. Structure du dépôt

### Structure actuelle

```text
sentinelx-ai/
├── data/
│   └── raw/                      # C-MAPSS FD001 (non versionné — voir § 7)
│       └── .gitkeep
├── notebooks/
│   └── 01_eda_fd001_final_fusion.ipynb   # EDA complète, exécutée (85 cellules, 17 sections)
├── artifacts/
│   └── eda_fd001/                # décisions de l'EDA (JSON) réutilisées par 02_baseline
├── reports/
│   └── figures/eda_fd001/        # 15 figures générées par le notebook
├── docs/
│   └── guide_eda_fd001.pdf       # guide pédagogique de l'EDA (36 pages)
├── requirements.txt
├── .gitignore
└── README.md
```

### Structure cible (cahier des charges)

```text
sentinelx-ai/
├── docker-compose.yml, .env.example
├── docs/            architecture/, report/, knowledge_base/
├── data/            README.md, sample/ (jamais les gros datasets)
├── notebooks/       01_eda.ipynb, 02_baseline.ipynb
├── ml/
│   ├── configs/
│   ├── src/         data.py, features.py, train_baseline.py, train_lstm.py, evaluate.py, inference.py
│   └── tests/
├── backend/app/     api/, models/, services/, rag/, simulator/  (+ tests/)
├── frontend/src/
├── infra/           postgres/, mlflow/
└── scripts/         download_data.py, seed_demo.py, run_demo.sh
```

Le cahier des charges précise : **les notebooks servent à l'exploration ; la logique finale doit être déplacée dans des scripts / modules** (`ml/src/`).

---

## 6. Installation

Prérequis : Python 3.14 (environnement testé avec Python 3.14.7), Git.

**Windows (PowerShell)**

```powershell
git clone <URL_DU_DEPOT>
cd sentinelx-ai
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS**

```bash
git clone <URL_DU_DEPOT>
cd sentinelx-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ouvrir ensuite le notebook dans VS Code (ou Jupyter) avec le noyau Python de `.venv`.

---

## 7. Données

### Source

Dataset **NASA C-MAPSS** (moteurs turbofan simulés jusqu'à la défaillance), sous-ensemble **FD001** : 1 condition opérationnelle, 1 mode de panne — le plus simple, recommandé pour le MVP. FD002 à FD004 (6 conditions et/ou 2 modes de panne) ne seront ajoutés qu'après stabilisation du pipeline.

- NASA — CMAPSS Jet Engine Simulated Data : <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
- NASA Prognostics Center of Excellence — Data Set Repository : <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>

### Mise en place

1. Télécharger le dataset et copier `train_FD001.txt`, `test_FD001.txt` et `RUL_FD001.txt` dans `data/raw/`, sans les modifier.
2. Vérifier l'intégrité des fichiers (empreintes ayant produit les résultats de l'EDA) :

| Fichier | SHA-256 |
|---|---|
| `train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |

```powershell
Get-FileHash data\raw\train_FD001.txt -Algorithm SHA256
```

Les données brutes ne sont **pas versionnées** (cahier des charges §18.1).

### Format

Fichiers texte sans en-tête, valeurs séparées par des espaces, 26 colonnes par ligne :

| Colonnes | Type | Signification |
|---|---|---|
| `unit_id` | entier | Identifiant du moteur (1 à 100 dans chaque fichier) |
| `cycle` | entier | Numéro du cycle, commence à 1, pas de 1 |
| `setting_1` à `setting_3` | réel | Condition opérationnelle |
| `sensor_1` à `sensor_21` | réel | Mesures de capteurs (non nommées dans les fichiers) |

- **Train** : 100 moteurs, 20 631 lignes, chaque trajectoire va jusqu'à la panne → RUL calculable.
- **Test** : 100 moteurs, 13 096 lignes, trajectoires interrompues avant la panne.
- **`RUL_FD001.txt`** : 100 valeurs = RUL réel au **dernier cycle observé** de chaque moteur de test. **Réservé à l'évaluation finale.**

Une ligne = l'état d'**un moteur à un cycle** : les 20 631 lignes du train sont 100 trajectoires temporelles, pas des exemples indépendants.

---

## 8. Analyse exploratoire (EDA) — travail réalisé

Notebook : [`notebooks/01_eda_fd001_final_fusion.ipynb`](notebooks/01_eda_fd001_final_fusion.ipynb) — 85 cellules, entièrement exécuté, sans erreur ni avertissement. Il commence par 9 questions (Q1 à Q9) auxquelles la conclusion répond.

### 8.1 Contenu du notebook

| § | Section | Ce qui est fait |
|---|---|---|
| 1 | Configuration et reproductibilité | Paramètres centralisés (graine 42, seuils), versions tracées, empreintes SHA-256, fonctions utilitaires |
| 2 | Schéma et chargement | Dictionnaire des colonnes, chargement train/test avec le même schéma |
| 3 | Qualité et intégrité | Manquants, infinis, doublons, unicité (moteur, cycle), continuité des cycles, démarrage à 1 |
| 4 | Structure des trajectoires | Longueurs de vie, percentiles, moteurs atypiques (règle de Tukey) |
| 5 | Split par moteur | `GroupShuffleSplit` 80/20 par `unit_id`, assertions anti-recouvrement, équilibre des longueurs |
| 6 | Variabilité | Classification constante / quasi constante / dynamique sur critères indépendants de l'échelle |
| 7 | Analyse temporelle | Trajectoires de 3 moteurs représentatifs (moyenne glissante causale) + profil de flotte en vie normalisée |
| 8 | Cible RUL | `RUL = T_i − t` sans colonne future, 5 contrôles automatiques, illustration du RUL plafonné |
| 9 | Distribution du RUL | Couverture des valeurs de RUL et nombre de moteurs par valeur |
| 10 | Corrélations | Pearson et Spearman avec le RUL, forme des relations |
| 11 | Robustesse par moteur | Spearman moteur par moteur, capteurs à deux régimes (`sensor_9`, `sensor_14`) |
| 12 | Redondance | Matrices Pearson / Spearman, paires fortes, regroupement hiérarchique |
| 13 | Audit anti-leakage | Tableau autorisé / interdit + assertions automatiques |
| 14 | Contrôle du test | Mêmes contrôles qualité, contrainte de fenêtre W, cohérence des distributions à stade de vie égal |
| 15 | Export | Artefacts JSON pour l'étape suivante |
| 16 | Synthèse | Résultats clés, décisions, couverture de F-02, **cellule qui vérifie automatiquement 27 chiffres cités** |
| 17 | Conclusion | Réponses à Q1-Q9, limites |

### 8.2 Résultats clés (valeurs réelles)

| Thème | Résultat |
|---|---|
| Qualité | Train et test : 0 manquant, 0 infini, 0 doublon, cycles continus démarrant à 1 |
| Trajectoires (train) | 128 à 362 cycles ; médiane 199 ; moyenne 206,31 ; écart-type 46,34 ; P5 = 147 ; P95 = 287,3 |
| Moteurs atypiques | 67 (313), 96 (336), 92 (341), 69 (362) au-delà de la borne de Tukey 307,6 — conservés |
| Split | 80 moteurs d'analyse (16 561 lignes) / 20 de validation (4 070 lignes), aucun recouvrement |
| Moteurs de validation | 1, 5, 11, 13, 19, 23, 31, 32, 34, 40, 45, 46, 54, 71, 74, 77, 78, 81, 84, 91 |
| Limite du split | Validation : longueurs 135 à 269 — aucun moteur de plus de 300 cycles |
| Variables constantes (7) | `setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19` |
| Quasi constante (1) | `sensor_6` : 21,61 (98,1 %) / 21,60 (1,9 %) |
| Dynamiques (16) | `setting_1`, `setting_2` + capteurs 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21 |
| Dégradation | Monotone, lente puis accélérée ; l'essentiel de la variation dans les 40 % finaux de la vie |
| RUL (analysis_train) | Moyenne 108,38 ; médiane 103 ; max 361 ; 60,87 % des lignes ≤ 125 ; tous les moteurs couvrent RUL 0 à 127 |
| Corrélations avec le RUL | `cycle` −0,78 (Spearman) ; 12 capteurs \|Spearman\| de 0,56 à 0,71 (max `sensor_11` −0,71) |
| Par moteur | 12 capteurs gardent le même signe sur 100 % des moteurs ; corrélation intra-moteur plus forte (ex. `sensor_11` −0,82) |
| Deux régimes | `sensor_9` (58 moteurs ρ < 0 / 22 ρ > 0), `sensor_14` (48 / 32) ; divergence sur les 50-100 derniers cycles |
| Redondance | 7 paires ≥ 0,80 ; `sensor_9`–`sensor_14` = 0,97 ; groupes {9, 14}, {8, 13}, {11, 12} ; bloc {4, 7, 11, 12} |
| Test | Tronqué : 31 à 303 cycles (médiane 133,5) ; **7 moteurs < 50 cycles** ; cohérent avec le train à stade de vie égal (\|SMD\| < 0,13) |

### 8.3 Décisions transmises au preprocessing

| # | Décision | Statut |
|---|---|---|
| 1 | Réutiliser le split figé `fd001_engine_split.json` | Acquis |
| 2 | Retirer les 7 variables constantes | Acquis |
| 3 | Ajuster scaler, imputation et sélection sur les moteurs d'entraînement uniquement | Acquis |
| 4 | Construire les fenêtres moteur par moteur, sans traverser deux trajectoires | Acquis |
| 5 | Fenêtre W ≤ 30 sans padding ; W = 50 exige une stratégie pour 7 moteurs de test | Contrainte |
| 6 | Évaluer par moteur et par tranche de RUL, en plus de MAE, RMSE, R² globaux | Acquis |
| 7 | Comparer RUL brut et RUL plafonné (plusieurs plafonds) | À valider |
| 8 | Tester `cycle` avec et sans | À valider |
| 9 | Ablation de `sensor_6`, `setting_1`, `setting_2` | À valider |
| 10 | Ablation de `sensor_9`, `sensor_14` ; gestion des groupes redondants selon le modèle | À valider |
| 11 | Tester des features causales relatives au moteur (écart au début de vie, pente récente) | Hypothèse |
| 12 | Comparer les modèles en validation croisée par groupes (`GroupKFold`) | Recommandé |

### 8.4 Artefacts (`artifacts/eda_fd001/`)

| Fichier | Clés principales |
|---|---|
| `fd001_engine_split.json` | `dataset`, `method`, `random_state` (42), `validation_size` (0.2), `analysis_engines` (80 ids), `validation_engines` (20 ids) |
| `fd001_feature_groups.json` | `near_constant_threshold`, `constant`, `near_constant`, `dynamic`, `heterogeneous_trend`, `redundant_groups`, `strong_pairs`, `forbidden_as_feature` |
| `eda_fd001_summary.json` | `raw_files_sha256`, `train`, `test` (dont `engines_shorter_than_window`), `rul_analysis_train`, `versions` |

Ces fichiers sont régénérés à l'identique à chaque exécution du notebook.

### 8.5 Documentation pédagogique

[`docs/guide_eda_fd001.pdf`](docs/guide_eda_fd001.pdf) — guide de 36 pages pour débutant, rédigé en **derja tunisienne avec les termes techniques en français** : bases de la maintenance prédictive, glossaire de 45 termes, statistiques, explication de chaque section du notebook avec ses résultats et figures, phrases prêtes pour le jury, 13 questions probables avec réponses en français, fiche mémo des chiffres.

---

## 9. Règles méthodologiques non négociables

Ces règles s'appliquent à **toutes** les étapes suivantes.

1. **Split par moteur** (`unit_id`), jamais par ligne aléatoire. Le split figé de l'EDA est réutilisé.
2. **`RUL_FD001.txt` n'est lu qu'une seule fois**, pour l'évaluation finale du pipeline figé. Il ne guide aucun choix (features, modèle, hyperparamètres, seuils).
3. **Variables interdites comme features** : `unit_id`, `RUL`, tout dernier cycle (`last_cycle`, `max_cycle`) et toute progression de vie (`life_pct`, `progress_pct`).
4. **Preprocessing ajusté sur le train uniquement** (scaler, imputation, sélection) puis appliqué à la validation et au test.
5. **Fenêtres temporelles par moteur**, sans chevaucher deux trajectoires ; features glissantes **causales** uniquement (passé seulement).
6. **Aucune métrique inventée** : chaque chiffre cité dans un notebook, le rapport ou le README doit provenir d'une exécution réelle. Pattern recommandé : une cellule finale qui revérifie par `assert` les chiffres cités dans le texte (comme la section 16 de l'EDA).
7. **Observation ≠ preuve** : une corrélation formule une hypothèse ; seule la validation des modèles la confirme. Aucune feature n'est supprimée sur la seule base d'une corrélation.
8. Les poids et seuils du Health Score sont des **choix produit documentés**, pas des vérités scientifiques.

---

## 10. Prochaine étape détaillée : 02_baseline

Objectif S3 (Definition of Done) : un baseline RUL entraîné, `baseline_metrics.json` et des graphiques, sans fuite d'information.

### 10.1 Charger les décisions de l'EDA

```python
import json
from pathlib import Path

import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
COLUMNS = ["unit_id", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]

train_df = pd.read_csv(ROOT / "data/raw/train_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
split = json.loads((ROOT / "artifacts/eda_fd001/fd001_engine_split.json").read_text(encoding="utf-8"))
groups = json.loads((ROOT / "artifacts/eda_fd001/fd001_feature_groups.json").read_text(encoding="utf-8"))

train = train_df[train_df["unit_id"].isin(split["analysis_engines"])].copy()
valid = train_df[train_df["unit_id"].isin(split["validation_engines"])].copy()

for df in (train, valid):
    df["RUL"] = df.groupby("unit_id")["cycle"].transform("max") - df["cycle"]

features = [c for c in COLUMNS[2:] if c not in groups["constant"]]   # 17 variables candidates
```

### 10.2 Plan pas à pas

1. **Déplacer la logique réutilisable dans `ml/src/`** : `data.py` (chargement, split, `add_rul`), `features.py` (sélection, fenêtres, features causales). Le notebook 02 importe ces modules.
2. **Cible** : comparer RUL brut et RUL plafonné (`RUL.clip(upper=cap)` avec plusieurs `cap`, par exemple 100 à 150). Le plafond est un hyperparamètre choisi en validation.
3. **Features** : partir des 17 variables non constantes ; ajouter ensuite des features causales par moteur (moyenne glissante, pente sur les W derniers cycles, écart au début de vie), avec W ≤ 30.
4. **Modèles baseline** : Ridge (interprétable, gère la redondance), Random Forest / Gradient Boosting (non linéaire) ; XGBoost en option.
5. **Validation** : `GroupKFold` par `unit_id` sur les 80 moteurs d'analyse pour comparer les configurations ; puis vérification sur les 20 moteurs de validation figés.
6. **Métriques** : MAE, RMSE, R² ; en bonus le score asymétrique NASA (pénalise davantage les prédictions tardives : `exp(−d/13) − 1` si `d < 0`, `exp(d/10) − 1` si `d ≥ 0`, avec `d = prédiction − réel`).
7. **Analyse d'erreur** : par moteur et par tranche de RUL (début, milieu, proche panne) ; tracer prédiction vs réel pour plusieurs moteurs.
8. **Ablations** : avec / sans `cycle` ; avec / sans `sensor_6`, `setting_1`, `setting_2` ; avec / sans `sensor_9`, `sensor_14` ; un capteur par groupe redondant.
9. **Évaluation finale sur le test** (une seule fois, pipeline figé) : prédire au **dernier cycle observé** de chaque moteur de test et comparer aux 100 valeurs de `RUL_FD001.txt`.
10. **Sorties** : `artifacts/baseline_fd001/baseline_metrics.json`, modèle sérialisé (non versionné s'il est lourd), graphiques dans `reports/figures/baseline_fd001/`.

### 10.3 Après le baseline (S4 — LSTM/GRU)

Fenêtres `W × F` construites par moteur ; W ∈ {20, 30} sans padding (W = 50 seulement avec une stratégie pour les 7 moteurs de test courts) ; normalisation ajustée sur le train ; dropout, early stopping ; comparaison au baseline sur le même split ; au moins 3 seeds si le temps le permet. Un modèle deep moins bon que le baseline est un résultat valide à analyser, pas à cacher.

---

## 11. Feuille de route complète (S4 → S12)

- **S5 — Anomalies + Health Score** : score d'anomalie simple et documenté (écart au comportement sain ou autoencoder), Health Score 0-100, seuils Warning / Critical réglés en validation, tests unitaires.
- **S6 — Backend + simulateur** : FastAPI (`/health`, `/machines`, `/simulations`, `/machines/{id}/predictions`, `/machines/{id}/alerts`, `/copilot/query`…), WebSocket `/ws/machines/{machine_id}`, schémas Pydantic, PostgreSQL (tables `machines`, `simulation_sessions`, `predictions`, `alerts`, `documents`, `chat_sessions`, `chat_messages`, `users`).
- **S7 — Dashboard MVP** : React + Vite — Fleet Overview, Machine Detail, Simulation Lab (start / pause / reset).
- **S8-S9 — RAG** : corpus de documents autorisés avec métadonnées (source, page, section), Qdrant, retrieval top-k, citations obligatoires, abstention (« Information insuffisante dans la base documentaire »), benchmark de 30 à 50 questions.
- **S10** : digital twin data-driven (2D d'abord), facteurs contributifs (XAI).
- **S11** : Docker Compose (frontend, backend, postgres, qdrant, mlflow), MLflow, tests d'intégration / E2E.
- **S12** : rapport, démo de 6 à 8 minutes, vidéo de secours, README portfolio avec résultats mesurés.

---

## 12. Reprendre le projet dans une nouvelle conversation

Cette section s'adresse à un assistant IA (ou à toute personne) qui reprend le projet sans historique.

### 12.1 Profil et préférences de la personne qui mène le projet

- **Premier projet en data science / IA** : elle veut **comprendre chaque étape** et pouvoir **défendre ses choix seule** devant le jury, pas seulement obtenir du code.
- **Langue des explications** : **derja tunisienne** (écrite en caractères latins), en gardant les **termes techniques en français ou en anglais** et en expliquant leur sens, avec des exemples simples.
- **Langue des livrables** (notebooks, rapport, README) : **français**, avec accents, ton professionnel.
- **Méthode attendue** : expliquer progressivement (petits groupes de cellules), montrer l'avant / après et le pourquoi de chaque changement, ne jamais inventer de résultat, exécuter le code et lire les sorties réelles avant d'écrire une interprétation.
- Quand elle le demande explicitement, accepter de tout faire en une fois, tout en gardant la même rigueur.
- Garder le **dossier propre** (pas de fichiers temporaires dans le projet), faire une copie de sauvegarde avant de modifier un notebook, et ne pas modifier l'environnement `.venv` sans le dire.

### 12.2 Fichiers à lire en premier

1. Ce `README.md` (contexte complet et décisions).
2. `notebooks/01_eda_fd001_final_fusion.ipynb`, en particulier les sections 13 (audit anti-leakage), 16 (synthèse et décisions) et 17 (conclusion).
3. `artifacts/eda_fd001/*.json` (split et groupes de variables à réutiliser).
4. `docs/guide_eda_fd001.pdf` pour le niveau d'explication attendu.

### 12.3 Prompt de reprise (à copier-coller)

```text
Je travaille sur le projet SentinelX AI (maintenance prédictive sur NASA C-MAPSS FD001).
Lis d'abord le README.md du dépôt : il contient tout le contexte, les décisions de l'EDA,
les règles anti-leakage et le plan de la prochaine étape.

Où j'en suis : l'EDA (notebooks/01_eda_fd001_final_fusion.ipynb) est terminée et validée.
Prochaine étape : S3 — notebook 02_baseline (baseline RUL) en suivant la section 10 du README.

Comment travailler avec moi :
- C'est mon premier projet dans le domaine : explique-moi chaque étape en derja tunisienne,
  avec les termes techniques en français/anglais et leur signification.
- Les notebooks et le rapport restent en français professionnel.
- N'invente jamais de résultat : exécute le code et base-toi sur les sorties réelles.
- Respecte les règles de la section 9 du README (split par moteur, RUL_FD001.txt lu une seule fois
  à la fin, preprocessing ajusté sur le train uniquement, aucune variable future).
- Avance étape par étape et explique pourquoi chaque choix est fait.
```

---

## 13. Environnement technique et pièges connus

- **Versions** (voir `requirements.txt`) : Python 3.14.7, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, ipykernel 7.4.0. Système de développement : Windows 11, VS Code, environnement `.venv` à la racine.
- **Chemins** : le notebook cherche `data/raw/` dans le dossier courant puis dans le dossier parent ; il fonctionne lancé depuis la racine ou depuis `notebooks/`.
- **pandas 3** : `DataFrame.stack()` conserve désormais les `NaN` → ajouter `.dropna()` après un `where(mask).stack()` ; affecter une valeur d'un type incompatible dans une colonne existante lève une erreur → construire les tableaux avec le bon type dès le départ.
- **Corrélation par moteur** : un capteur constant sur un moteur donne une corrélation indéfinie (`ConstantInputWarning`) → exclure explicitement ces moteurs (déjà fait en section 11 de l'EDA).
- **Matplotlib hors notebook** : utiliser le backend `Agg` pour une exécution non interactive.
- **Données** : ne jamais versionner `data/raw/` ; ne jamais committer de secrets (`.env` est ignoré ; seul un `.env.example` sans secret sera versionné).

---

## 14. Limites actuelles

- Données **simulées** : les conclusions décrivent le simulateur C-MAPSS, pas un parc réel.
- Un seul sous-ensemble (FD001) : une condition opérationnelle, un mode de panne.
- Validation de 20 moteurs seulement, sans trajectoire de plus de 269 cycles.
- Les capteurs ne sont pas nommés dans les fichiers : aucune interprétation physique n'est faite.
- Aucun modèle n'est encore entraîné : aucune métrique de performance n'existe à ce stade.

---

## 15. Références

- A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, International Conference on Prognostics and Health Management (PHM), 2008.
- NASA — CMAPSS Jet Engine Simulated Data : <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
- Documentation : [FastAPI](https://fastapi.tiangolo.com/), [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/), [Qdrant — hybrid search & reranking](https://qdrant.tech/documentation/tutorials-basics/reranking-hybrid-search/), [Docker Compose](https://docs.docker.com/compose/).
