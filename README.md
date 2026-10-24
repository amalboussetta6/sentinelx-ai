# SentinelX AI — Maintenance prédictive & Maintenance Copilot RAG

**Prédire la durée de vie utile restante (RUL) de moteurs d'avion à partir de leurs capteurs, puis l'intégrer dans une plateforme de suivi assistée par un copilote documentaire.**

> Projet d'intégration en Intelligence Artificielle (4e année du cycle ingénieur en IA). Prototype académique d'aide à la décision, **100 % logiciel** : des données publiques de la NASA sont rejouées comme un flux temps réel, sans capteur physique.

| | |
|---|---|
| **Avancement** | S2 (EDA) et S3 (baseline RUL) terminés et vérifiés · prochaine étape : **S4 — LSTM / GRU** |
| **Résultat actuel** | Forêt aléatoire : **RMSE = 12,8 cycles** en validation croisée, **11,4** sur 20 moteurs de validation, **14,4** sur le test NASA C-MAPSS FD001 (test déjà consulté par des versions précédentes, voir [section 6](#6-baseline-rul-s3)) |
| **Outils** | Python 3.14, pandas, NumPy, scikit-learn, matplotlib, seaborn, pytest |
| **Documentation** | [Guide de l'EDA](docs/guide_eda.md) · [Guide de la baseline](docs/guide_baseline.md) · [Cahier des charges](docs/cahier_des_charges_sentinelx_ai.pdf) |
| **Dernière mise à jour** | 24 octobre 2026 |

---

## Sommaire

1. [Présentation et objectifs](#1-présentation-et-objectifs)
2. [Données](#2-données)
3. [Architecture : ce qui existe, ce qui est prévu](#3-architecture--ce-qui-existe-ce-qui-est-prévu)
4. [État d'avancement](#4-état-davancement)
5. [Analyse exploratoire (S2)](#5-analyse-exploratoire-s2)
6. [Baseline RUL (S3)](#6-baseline-rul-s3)
7. [Règles méthodologiques](#7-règles-méthodologiques)
8. [Structure du dépôt](#8-structure-du-dépôt)
9. [Installation et exécution](#9-installation-et-exécution)
10. [Fichiers produits](#10-fichiers-produits)
11. [Prochaine étape : S4 — LSTM / GRU](#11-prochaine-étape--s4--lstm--gru)
12. [Feuille de route](#12-feuille-de-route)
13. [Limites et perspectives](#13-limites-et-perspectives)
14. [Références](#14-références)

---

## 1. Présentation et objectifs

Une panne non anticipée immobilise un équipement et perturbe la production. Les capteurs contiennent des signes de dégradation, mais ils sont bruités et difficiles à interpréter. SentinelX AI vise à transformer ces mesures en **indicateurs utiles** : durée de vie restante (RUL), score d'anomalie, Health Score et alertes. Un **copilote RAG** répondra aux questions de maintenance en citant ses sources.

> *Problématique : comment concevoir une plateforme d'IA capable d'anticiper la dégradation d'équipements à partir de séries temporelles industrielles, de simuler leur suivi en temps réel et d'assister la maintenance grâce à un copilote RAG fiable et sourcé ?*

**Objectifs** ([cahier des charges](docs/cahier_des_charges_sentinelx_ai.pdf), §1.3) :

- préparer les séries temporelles de façon reproductible ;
- établir une **baseline classique** avant le deep learning *(fait)* ;
- prédire le RUL sur C-MAPSS FD001 *(baseline faite ; LSTM/GRU en S4)* ;
- détecter les anomalies et calculer un Health Score ;
- exposer les modèles via une API, simuler l'arrivée des données, afficher un dashboard ;
- interroger une base documentaire avec un copilote RAG sourcé ;
- conteneuriser l'ensemble.

## 2. Données

**NASA C-MAPSS** simule des moteurs d'avion qui fonctionnent jusqu'à la panne. Le sous-ensemble **FD001** (une condition de fonctionnement, un type de panne) est le plus simple, et c'est le point de départ recommandé.

| Fichier | Contenu |
|---|---|
| `train_FD001.txt` | 100 moteurs suivis **jusqu'à la panne** (20 631 lignes) |
| `test_FD001.txt` | 100 moteurs **arrêtés avant la panne** (13 096 lignes) |
| `RUL_FD001.txt` | vrai RUL de chaque moteur de test : **réservé à l'évaluation finale** |

Chaque ligne décrit **un moteur à un cycle** : `unit_id`, `cycle`, 3 réglages (`setting_1` à `setting_3`) et 21 capteurs anonymes (`sensor_1` à `sensor_21`). La cible est **RUL = dernier cycle du moteur − cycle courant**.

**Installation des données.** Les données ne sont pas versionnées dans Git. Il faut télécharger le dataset depuis la [page NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) et copier les trois fichiers, sans les modifier, dans `data/raw/`. Les notebooks vérifient leur empreinte SHA-256 :

| Fichier | SHA-256 |
|---|---|
| `train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |

## 3. Architecture : ce qui existe, ce qui est prévu

```text
 Données C-MAPSS ──► Préparation (EDA, features) ──► Modèles (RUL, anomalies) ──► Health Score / alertes
        │                                                   │
        └──► Simulateur (rejoue les cycles) ──► API FastAPI ──► PostgreSQL · Dashboard React
                                                    └──► Copilote RAG (Qdrant + LLM, citations)
                          Déploiement : Docker Compose · Suivi des expériences : MLflow
```

| Composant | Technologie | État |
|---|---|---|
| Analyse exploratoire | pandas, matplotlib, seaborn | **Fait** (`notebooks/01_eda_fd001.ipynb`) |
| Préparation des données et features | pandas, NumPy (`ml/src/`) | **Fait et testé** (`pytest`) |
| Baseline RUL | scikit-learn | **Fait** (`notebooks/02_baseline_fd001.ipynb`) |
| Modèle séquentiel LSTM / GRU | PyTorch | Prévu (S4) |
| Anomalies + Health Score | Python | Prévu (S5) |
| API + simulateur + base de données | FastAPI, PostgreSQL | Prévu (S6) |
| Dashboard | React + Vite | Prévu (S7) |
| Copilote RAG | Qdrant + LLM | Prévu (S8-S9) |
| Digital twin, explicabilité | React / Three.js | Prévu (S10, optionnel) |
| Docker Compose, MLflow, tests d'intégration | — | Prévu (S11) |

## 4. État d'avancement

| Semaine | Étape | État vérifié |
|---|---|---|
| S1 | Cadrage + dépôt | **Partiel** : dépôt, environnement, documentation. Pas encore de `docker-compose.yml`, `.env.example` ni `scripts/`. |
| **S2** | **Données + EDA** | **Terminé** : notebook exécuté sans erreur, 3 fichiers JSON de décisions, 8 figures. |
| **S3** | **Baseline RUL** | **Terminé** : 2 modèles comparés par validation croisée par moteur (5 tours), vérification sur 20 moteurs de validation, test lu une fois dans le notebook, 8 figures, 29 tests `pytest` réussis. |
| S4 | LSTM / GRU | Prochaine étape |

*Vérification du 24 octobre 2026* : les deux notebooks ont été exécutés de bout en bout, dans l'ordre, avec les versions de `requirements.txt` ; chacun se termine par une cellule qui revérifie les chiffres cités dans son texte.

## 5. Analyse exploratoire (S2)

Notebook : [`notebooks/01_eda_fd001.ipynb`](notebooks/01_eda_fd001.ipynb) · Guide : [`docs/guide_eda.md`](docs/guide_eda.md)

Pour chaque étape, le notebook explique ce que nous faisons, pourquoi, comment fonctionne le code, ce que nous obtenons et ce que nous en concluons.

**Principaux constats**

- Données **propres** : aucune valeur manquante, aucun doublon, cycles continus.
- Durées de vie de **128 à 362 cycles**.
- **14 capteurs retenus** : 7 colonnes constantes, `sensor_6` (presque constant) et les réglages `setting_1/2` (sans lien avec le RUL) sont écartés.
- Dégradation **lente puis accélérée**, avec un **niveau de départ propre à chaque moteur**.
- 12 capteurs nettement liés au RUL (corrélation de Spearman de 0,56 à 0,71 en valeur absolue).
- Le test a des moteurs de 31 cycles seulement → **historique de 30 cycles au maximum**.

## 6. Baseline RUL (S3)

Notebook : [`notebooks/02_baseline_fd001.ipynb`](notebooks/02_baseline_fd001.ipynb) · Guide : [`docs/guide_baseline.md`](docs/guide_baseline.md)

**Méthode**

| Étape | Choix |
|---|---|
| Données | 80 moteurs d'entraînement et 20 de validation, repris de l'EDA |
| Features | pour chacun des 14 capteurs : moyenne des 30 derniers cycles, pente sur 30 cycles, écart au début de vie. Soit 42 features, calculées avec le passé uniquement. |
| Modèles | régression linéaire ; forêt aléatoire (100 arbres, `min_samples_leaf` = 5, `max_features` = 0,33) |
| Choix | **validation croisée par moteur** (`GroupKFold`, 5 tours de 64 moteurs / 16 moteurs) sur les 80 moteurs d'entraînement ; erreur mesurée sur la zone RUL réel ≤ 127. Trois expériences : mesures instantanées ou historique, plafond du RUL, `max_features` de la forêt. Règle : le plus petit RMSE moyen gagne. |
| Vérification | le modèle choisi, entraîné sur les 80 moteurs, est évalué **une fois** sur les 20 moteurs de validation, qui n'ont servi à aucun choix |
| Test | réentraînement sur les 100 moteurs ; prédiction au **dernier cycle observé** des 100 moteurs de test ; `RUL_FD001.txt` lu une seule fois dans le notebook |

**Résultats** (fichier `artifacts/baseline_fd001/baseline_metrics.json`)

| Modèle (RUL plafonné à 110) | RMSE validation croisée (80 moteurs) | RMSE validation (20 moteurs) | **RMSE test** | MAE test | R² test |
|---|---|---|---|---|---|
| Régression linéaire | 16,8 | 15,7 | 19,1 | 15,1 | 0,79 |
| **Forêt aléatoire (retenue)** | **12,8** | **11,4** | **14,4** | **10,5** | **0,88** |

- RMSE et MAE sont en cycles (plus bas = meilleur). R² vaut 1 pour une prédiction parfaite.
- L'historique fait passer le RMSE de la forêt de 20,7 à 13,8 en validation croisée (plafond 125). Sans plafond, il monte à 23,7.
- Près de la panne (RUL ≤ 25), la forêt se trompe en moyenne de **2,8 cycles** sur les 20 moteurs de validation.
- Sur le test, 51 moteurs sur 100 reçoivent une prédiction un peu trop optimiste (au plus 17,8 cycles).

**Pourquoi la validation et le test donnent des RMSE différents.** La validation évalue **toutes les lignes** dont le RUL réel est ≤ 127, sur des moteurs suivis jusqu'à la panne. Le test NASA n'évalue **qu'un point par moteur** (son dernier cycle observé), avec des RUL de 7 à 145 : 28 moteurs dépassent le plafond de 110 et sont volontairement sous-estimés, et 7 moteurs ont moins de 50 cycles d'historique.

**Utilisation honnête du jeu de test.** `RUL_FD001.txt` a déjà été lu par des versions précédentes de la baseline (RMSE 22,69 puis 14,23), puis par un prototype simplifié (17,05). Le test n'a jamais servi directement à un choix : tous les choix suivent une règle écrite à l'avance (plus petit RMSE moyen en validation croisée). Mais connaître ces scores a pu orienter le travail : **le RMSE test de 14,4 n'est pas une évaluation totalement indépendante** et il est probablement un peu optimiste. Les scores de validation croisée et des 20 moteurs de validation sont les références les plus propres. Détails : section 8 du notebook et [guide de la baseline](docs/guide_baseline.md#5-utilisation-honnête-du-jeu-de-test).

## 7. Règles méthodologiques

Ces règles s'appliquent à toutes les étapes, y compris S4.

1. **Séparer par moteur**, jamais par ligne, en réutilisant `artifacts/eda_fd001/fd001_engine_split.json`.
2. **Lire `RUL_FD001.txt` une seule fois**, à la fin, pour l'évaluation finale. Il ne guide aucun choix.
3. **Ne jamais utiliser comme entrée** `unit_id`, `RUL`, le dernier cycle du moteur ou le pourcentage de vie écoulé.
4. **Calculer les features avec le passé uniquement**, moteur par moteur.
5. **Faire les choix par validation croisée** sur les 80 moteurs d'entraînement (mêmes 5 groupes : `ml.src.evaluation.engine_folds`). Les 20 moteurs de validation vérifient une fois le modèle choisi. Tout prétraitement appris (par exemple une normalisation pour le LSTM) est calculé sur les moteurs d'entraînement uniquement.
6. **Ne jamais inventer de chiffre** : chaque notebook revérifie automatiquement les chiffres cités dans son texte.
7. **Une corrélation est une piste**, pas une preuve : seule la validation confirme l'utilité d'une variable.

## 8. Structure du dépôt

```text
sentinelx-ai/
├── data/raw/                       # fichiers C-MAPSS FD001 (non versionnés ; seul .gitkeep est suivi)
├── notebooks/
│   ├── 01_eda_fd001.ipynb          # S2 — analyse exploratoire
│   └── 02_baseline_fd001.ipynb     # S3 — baseline RUL
├── ml/
│   ├── src/                        # fonctions réutilisables : mêmes noms et mêmes calculs que le notebook de baseline
│   │   ├── data.py                 # chargement des fichiers, calcul du RUL, dernier cycle par moteur
│   │   ├── features.py             # rolling_mean, rolling_slope, delta_from_start, add_history_features (passé uniquement)
│   │   ├── models.py               # régression linéaire et forêt aléatoire
│   │   └── evaluation.py           # compute_metrics, evaluate_on_zone, engine_folds (5 groupes de moteurs)
│   └── tests/                      # 29 tests pytest
├── artifacts/
│   ├── eda_fd001/                  # décisions de l'EDA (3 fichiers JSON)
│   └── baseline_fd001/             # métriques, prédictions du test, modèle .joblib (non versionné)
├── reports/figures/
│   ├── eda_fd001/                  # 8 figures de l'EDA
│   └── baseline_fd001/             # 8 figures de la baseline
├── docs/
│   ├── guide_eda.md                # résumé de l'EDA et préparation de la soutenance
│   ├── guide_baseline.md           # résumé de la baseline et préparation de la soutenance
│   ├── cahier_des_charges_sentinelx_ai.pdf
│   └── reprise_projet.md           # consignes pour reprendre le projet
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. Installation et exécution

**Prérequis** : Python 3.14, Git, et les fichiers FD001 dans `data/raw/` (section 2).

```powershell
# Windows (PowerShell)
git clone https://github.com/amalboussetta6/sentinelx-ai.git
cd sentinelx-ai
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
git clone https://github.com/amalboussetta6/sentinelx-ai.git
cd sentinelx-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Exécuter les notebooks dans cet ordre** (VS Code ou Jupyter, avec le noyau Python de `.venv`) :

| Ordre | Notebook | Durée indicative |
|---|---|---|
| 1 | `notebooks/01_eda_fd001.ipynb` | moins d'une minute |
| 2 | `notebooks/02_baseline_fd001.ipynb` | environ 7 à 8 minutes (validation croisée) |

Le notebook 2 lit les décisions écrites par le notebook 1 : il faut donc lancer l'EDA en premier sur une installation neuve.

**Tests** (depuis la racine du projet) : `python -m pytest`. Les tests sur les données réelles et les artefacts sont ignorés si `data/raw/` ou `artifacts/baseline_fd001/` sont vides : lancer les notebooks d'abord.

## 10. Fichiers produits

| Fichier | Produit par | Contenu |
|---|---|---|
| `artifacts/eda_fd001/fd001_engine_split.json` | EDA | moteurs d'entraînement (80) et de validation (20) |
| `artifacts/eda_fd001/fd001_feature_groups.json` | EDA | colonnes constantes, colonnes écartées, 14 capteurs retenus, paires redondantes, variables interdites |
| `artifacts/eda_fd001/eda_fd001_summary.json` | EDA | empreintes des fichiers, chiffres clés, zone d'évaluation (127), historique (30 cycles) |
| `artifacts/baseline_fd001/baseline_metrics.json` | baseline | choix retenus, moteurs des 5 tours de validation croisée, métriques de validation et de test des deux modèles, historique des lectures du test : **référence pour S4** |
| `artifacts/baseline_fd001/test_predictions.csv` | baseline | RUL réel et prédit des 100 moteurs de test |
| `artifacts/baseline_fd001/baseline_rul_fd001.joblib` | baseline | modèle retenu et ses réglages (non versionné, utile pour l'API en S6) |
| `reports/figures/*/*.png` | notebooks | 16 figures pour le rapport |

## 11. Prochaine étape : S4 — LSTM / GRU

Objectif : vérifier si un modèle séquentiel (un réseau de neurones qui lit une suite de cycles) fait **mieux** que la baseline.

- Mêmes 5 tours de validation croisée sur les 80 moteurs d'entraînement pour tous les choix (taille de fenêtre, architecture, nombre d'époques…).
- Mêmes 20 moteurs de validation, utilisés une seule fois pour vérifier le modèle choisi.
- Même zone d'évaluation (RUL réel ≤ 127) et RUL plafonné.
- Fenêtres de 30 cycles au maximum, construites moteur par moteur.
- **Normalisation** calculée sur les moteurs d'entraînement uniquement (dans chaque tour de validation croisée).
- Test au dernier cycle observé, avec une seule lecture de `RUL_FD001.txt`, **à la toute fin**.
- **Scores à battre : RMSE 12,8 en validation croisée, 11,4 sur les 20 moteurs de validation, 14,4 sur le test.**
- Les fonctions de `ml/src/` (chargement, RUL, métriques, `evaluate_on_zone`, `engine_folds`) peuvent être réutilisées.
- **Représentation différente, protocole identique.** Le LSTM/GRU recevra directement des fenêtres de mesures brutes des 14 capteurs, et non les 42 features calculées à la main pour la forêt aléatoire. Ce qui reste commun : les données, le RUL, les moteurs, les 5 groupes, les métriques et la zone d'évaluation.

Si le LSTM ne fait pas mieux, c'est un résultat valable, qu'il faudra analyser et présenter honnêtement.

## 12. Feuille de route

| Semaine | Étape | État |
|---|---|---|
| S1 | Cadrage + dépôt | Partiel |
| S2 | Données + EDA | **Terminé** |
| S3 | Baseline RUL | **Terminé** |
| S4 | LSTM / GRU + rapport comparatif | Prochaine étape |
| S5 | Anomalies + Health Score | À faire |
| S6 | API FastAPI + simulateur + PostgreSQL | À faire |
| S7 | Dashboard (MVP démontrable) | À faire |
| S8-S9 | Copilote RAG (corpus, Qdrant, citations) | À faire |
| S10 | Digital twin + explicabilité | À faire (optionnel) |
| S11 | Docker Compose + MLflow + tests | À faire |
| S12 | Rapport + soutenance | À faire |

## 13. Limites et perspectives

**Limites actuelles**

- Données **simulées**, une seule condition de fonctionnement et un seul type de panne (FD001).
- Peu de moteurs : 80 pour les choix, 20 pour la vérification.
- Le jeu de test a déjà été consulté par des versions précédentes : son score n'est pas totalement indépendant (section 6).
- La baseline regarde 30 cycles d'historique, ne distingue pas les moteurs au-delà de 110 cycles restants, et ne donne pas d'intervalle de confiance.
- Aucune brique applicative (API, dashboard, RAG) n'est encore réalisée.

**Perspectives** : modèle séquentiel en S4 ; prise en compte des prédictions trop optimistes dans les seuils d'alerte (S5) ; extension à FD002–FD004 une fois le pipeline stabilisé.

## 14. Références

- A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, International Conference on Prognostics and Health Management (PHM), 2008.
- NASA — CMAPSS Jet Engine Simulated Data : <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
- Documentation : [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/docs/), [FastAPI](https://fastapi.tiangolo.com/), [Docker Compose](https://docs.docker.com/compose/).
