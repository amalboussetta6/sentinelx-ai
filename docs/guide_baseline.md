# Guide de la baseline RUL — NASA C-MAPSS FD001

Résumé du notebook [`notebooks/02_baseline_fd001.ipynb`](../notebooks/02_baseline_fd001.ipynb), pour réviser et préparer la soutenance. Il fait suite au [guide de l'EDA](guide_eda.md).

## 1. À quoi sert la baseline ?

Une **baseline** est un premier modèle simple et bien évalué. Elle donne le score à battre : le LSTM/GRU de l'étape S4 n'aura d'intérêt que s'il fait mieux, évalué de la même façon.

## 2. La méthode en 6 étapes

| Étape | Ce que nous faisons |
|---|---|
| 1. Données | reprendre les 80 / 20 moteurs et les 14 capteurs de l'EDA ; calculer le RUL |
| 2. Features | pour chaque capteur : **moyenne** des 30 derniers cycles, **pente** sur 30 cycles, **écart au début de vie** (42 features au total), calculées avec le passé uniquement |
| 3. Modèles | **régression linéaire** (somme pondérée des features) et **forêt aléatoire** (moyenne de 100 arbres de décision) |
| 4. Choix par validation croisée | sur les 80 moteurs d'entraînement, 5 tours (64 moteurs pour apprendre, 16 pour contrôler) ; erreur mesurée sur la zone RUL réel ≤ 127 ; le plus petit RMSE moyen gagne |
| 5. Vérification | le modèle choisi, entraîné sur les 80 moteurs, est évalué **une fois** sur les 20 moteurs de validation |
| 6. Test final | réentraîner sur les 100 moteurs, prédire le RUL au dernier cycle observé des 100 moteurs de test, lire `RUL_FD001.txt` une seule fois |

### Les trois groupes de moteurs

| Groupe | Rôle |
|---|---|
| 80 moteurs d'entraînement | le modèle apprend ; **tous les choix** sont faits ici, par validation croisée |
| 20 moteurs de validation | jamais utilisés pour choisir : ils vérifient une fois le modèle choisi |
| 100 moteurs de test | examen final officiel de la NASA |

Image simple : on révise avec des exercices (entraînement), on vérifie son niveau sur des exercices corrigés qu'on n'a pas utilisés pour réviser (validation), puis on passe l'examen (test).

### La validation croisée par moteur (`GroupKFold`)

Avec un seul petit groupe de contrôle, une option pourrait gagner par chance. La validation croisée répartit les 80 moteurs en **5 groupes de 16**. À chaque tour, le modèle apprend sur 4 groupes et il est contrôlé sur le 5e. Chaque moteur est contrôlé une fois, par un modèle qui ne l'a jamais vu. Le score d'une option est la **moyenne des 5 tours** ; l'écart-type indique si le résultat est stable.

`GroupKFold` garde **toutes les lignes d'un moteur dans le même groupe** : un moteur n'est jamais à la fois en apprentissage et en contrôle. Dans le notebook, c'est une simple boucle `for` sur les 5 tours. `GroupKFold` renvoie des positions de lignes (`fit_index`, `check_index`) ; le notebook les transforme tout de suite en **listes de numéros de moteurs** (avec `.iloc`), puis sélectionne les lignes avec `isin`. Les 5 listes de moteurs contrôlés sont affichées dans le notebook et enregistrées dans `baseline_metrics.json`.

### Les fonctions du notebook (et de `ml/src/`)

Les mêmes fonctions, avec les mêmes noms, existent dans le notebook et dans `ml/src/` ; le notebook (section 11) et les tests vérifient qu'elles donnent exactement les mêmes résultats.

| Fonction | Rôle |
|---|---|
| `rolling_mean` | moyenne des 30 derniers cycles (`rolling(30, min_periods=1)`) |
| `compute_slope`, `rolling_slope` | pente des 30 derniers cycles, par une boucle sur les cycles et `np.polyfit` (0 au premier cycle) |
| `delta_from_start` | mesure − référence : moyenne des cycles déjà vus avant le cycle 20 (`expanding().mean()`), puis moyenne figée des 20 premiers cycles |
| `add_history_features` | boucle sur les moteurs, appelle les trois fonctions pour chaque capteur |
| `create_model` | crée une régression linéaire ou une forêt aléatoire neuve |
| `train_and_predict` | entraîne sur un groupe de moteurs (RUL plafonné si un plafond est donné) et prédit sur un autre |
| `evaluate_on_zone` | MAE, RMSE et R² sur les lignes dont le RUL réel est ≤ 127 |
| `cross_validate` | les 5 tours de validation croisée, puis la moyenne |

Un exemple sur un **moteur fictif de 5 cycles** (fenêtre de 3, début de vie de 2) montre, dans le notebook, les calculs faits à la main et retrouvés par le code.

### Deux façons de mesurer l'erreur

| | Validation (croisée et 20 moteurs) | Test NASA |
|---|---|---|
| Moteurs | suivis jusqu'à la panne | arrêtés avant la panne |
| Lignes évaluées | toutes les lignes avec un RUL réel ≤ 127 | une seule par moteur : son **dernier cycle observé** |
| RUL évalués | de 0 à 127 | de 7 à 145 |

La validation mesure le modèle tout au long de la fin de vie. Le test reproduit la question réelle : « ce moteur a tourné jusqu'ici, combien lui reste-t-il ? ».

### Les métriques

| Métrique | Lecture |
|---|---|
| MAE | erreur moyenne, en cycles |
| RMSE | erreur en cycles où les grosses erreurs comptent davantage : la métrique principale |
| R² | 1 = parfait, 0 = pas mieux que prédire toujours la moyenne |

## 3. Résultats

**Expérience 1 — l'historique aide-t-il ?** (RMSE moyen en validation croisée, plafond 125)

| | 14 mesures instantanées | 42 features d'historique |
|---|---|---|
| Régression linéaire | 21,6 | 18,2 |
| Forêt aléatoire | 20,7 | **13,8** |

**Expérience 2 — quel plafond ?** (RMSE moyen en validation croisée, features d'historique)

| Plafond | sans | 100 | 110 | 125 | 140 |
|---|---|---|---|---|---|
| Régression linéaire | 32,5 | 17,1 | **16,8** | 18,2 | 20,7 |
| Forêt aléatoire | 23,7 | 13,5 | **12,8** | 13,8 | 15,7 |

**Expérience 3 — `max_features` de la forêt** : 0,33 → **12,85** ; 1,0 → 13,16. Nous gardons 0,33.

**Résultat final.** Ces chiffres viennent de `artifacts/baseline_fd001/baseline_metrics.json`.

| Modèle (plafond 110) | RMSE validation croisée | RMSE validation (20 moteurs) | RMSE test | MAE test | R² test |
|---|---|---|---|---|---|
| Régression linéaire | 16,8 | 15,7 | 19,1 | 15,1 | 0,79 |
| **Forêt aléatoire (retenue)** | **12,8** | **11,4** | **14,4** | **10,5** | **0,88** |

**Ce qu'il faut retenir.**

- L'**historique récent** réduit fortement l'erreur, surtout pour la forêt aléatoire.
- **Plafonner le RUL** est indispensable. 110 donne le meilleur résultat, mais 100 et 125 sont proches.
- Près de la panne (RUL ≤ 25), la forêt se trompe en moyenne de **2,8 cycles** sur les 20 moteurs de validation.
- Elle s'appuie surtout sur les moyennes (49 % de l'importance) et les écarts au début de vie (42 %).
- Sur le test, 51 moteurs sur 100 reçoivent une prédiction un peu trop optimiste, au plus de 17,8 cycles.
- Le test est moins bon que la validation pour trois raisons :
  - 28 moteurs de test sont encore sains (RUL > 110), donc hors de portée du plafond : sous-estimés de 19,4 cycles en moyenne ;
  - 7 moteurs ont moins de 50 cycles d'historique (erreur moyenne 14,0 contre 10,3) ;
  - le test ne mesure qu'un point par moteur, avec des RUL jusqu'à 145.

## 4. Limites

- L'historique est limité à 30 cycles, résumés par des statistiques choisies à la main.
- Au-delà de 110 cycles, le modèle ne distingue plus un moteur neuf d'un moteur à mi-vie.
- Il annonce parfois un peu trop de vie restante près de la panne.
- Il donne une valeur sans intervalle de confiance.
- Peu de moteurs (80 pour les choix, 20 pour la vérification), et les données sont simulées.
- Le jeu de test a déjà été consulté (section suivante).

## 5. Utilisation honnête du jeu de test

En principe, `RUL_FD001.txt` ne doit être lu qu'une fois. Dans ce projet, il a été lu par plusieurs versions de la baseline :

| Version | Modèle | RMSE test | Comment les choix ont été faits |
|---|---|---|---|
| Version 1 | Ridge, mesures du cycle courant | 22,69 | validation croisée |
| Version 2 | forêt aléatoire, 62 features | 14,23 | validation croisée |
| Prototype simplifié | forêt aléatoire, 42 features (variante proche), `max_features` = 1,0 | 17,05 | aucun : prototype |
| **Version actuelle** | forêt aléatoire, 42 features, `max_features` = 0,33 | **14,40** | validation croisée |

Le test n'a jamais servi directement à un choix. Mais connaître le score du prototype a orienté la suite du travail. **Le RMSE test de 14,4 n'est donc pas une évaluation totalement indépendante** : il est probablement un peu optimiste.

**Ce qui limite ce biais :**

1. Une **règle écrite à l'avance** : tous les choix (features, plafond, modèle, `max_features`) vont au plus petit RMSE moyen en validation croisée sur les 80 moteurs d'entraînement.
2. Les **20 moteurs de validation** donnent une seconde estimation qui n'a servi à aucun choix (11,4).
3. **Pour S4**, le protocole est fixé avant de commencer : choix uniquement par validation croisée, test lu une seule fois, à la toute fin.

Une évaluation parfaitement indépendante demanderait des données jamais consultées pendant le projet.

## 6. Référence pour S4 (LSTM / GRU)

Le modèle séquentiel doit être évalué **de la même façon** :

- mêmes 5 tours de validation croisée sur les 80 moteurs (`fold_engines` dans `baseline_metrics.json`, ou `ml.src.evaluation.engine_folds`) pour tous les choix ;
- mêmes 20 moteurs de validation, utilisés une fois ;
- zone RUL réel ≤ 127 et RUL plafonné ;
- 30 cycles d'historique au maximum ;
- test au dernier cycle observé avec une seule lecture de `RUL_FD001.txt`, à la fin.

**Scores à battre : RMSE 12,8 en validation croisée, 11,4 sur les 20 moteurs de validation, 14,4 sur le test.**

Les fonctions de chargement, de RUL, de métriques (`evaluate_on_zone`) et de groupes de moteurs (`engine_folds`) sont dans `ml/src/`. Le LSTM recevra des **fenêtres de mesures brutes** des capteurs, pas les 42 features de la forêt : la représentation change, le protocole de comparaison reste le même. Contrairement à la baseline, il aura besoin d'une **normalisation**, calculée sur les moteurs d'apprentissage uniquement (dans chaque tour de validation croisée).

## 7. Préparation de la soutenance

### Résumé en une minute

> « Après l'EDA, j'ai construit une baseline pour prédire le RUL. Pour chaque capteur, j'ai calculé trois features qui décrivent l'historique récent du moteur : la moyenne des 30 derniers cycles, la pente et l'écart au début de vie, en utilisant uniquement le passé. J'ai comparé une régression linéaire et une forêt aléatoire. Tous mes choix sont faits par validation croisée par moteur sur 80 moteurs ; 20 autres moteurs vérifient le modèle choisi. L'historique améliore nettement les résultats, et le meilleur plafond de RUL est 110 cycles. La forêt aléatoire est retenue : RMSE de 12,8 en validation croisée, 11,4 sur les 20 moteurs de validation et 14,4 sur le test NASA. Ce dernier score est un peu optimiste, car le test avait déjà été consulté par des versions précédentes. Ce sont les scores que le LSTM devra battre. »

### Questions probables du jury

| Question | Réponse |
|---|---|
| Pourquoi une baseline avant le deep learning ? | Pour avoir un score de référence. Sans elle, on ne peut pas savoir si un LSTM apporte un vrai gain. |
| Pourquoi ces deux modèles ? | La régression linéaire est le modèle le plus simple et le plus facile à expliquer. La forêt aléatoire peut représenter une relation non linéaire, comme une dégradation qui s'accélère. |
| Pourquoi plafonner le RUL ? | En début de vie, les capteurs bougent très peu : il est très difficile de distinguer 250 et 300 cycles restants. Sans plafond, le RMSE de la forêt passe de 12,8 à 23,7 en validation croisée. |
| Comment comparer des plafonds différents ? | Le plafond ne s'applique qu'à l'apprentissage. L'évaluation compare toujours au **vrai** RUL. |
| Pourquoi une validation croisée ? | Avec un seul groupe de 20 moteurs, une option pourrait gagner par chance. La moyenne de 5 tours est plus fiable, et l'écart-type montre la stabilité. |
| Pourquoi `GroupKFold` et pas `KFold` ? | `KFold` mélangerait les lignes d'un même moteur entre apprentissage et contrôle : le modèle reconnaîtrait le moteur, et le score serait trop optimiste. |
| À quoi servent alors les 20 moteurs de validation ? | À vérifier une fois le modèle choisi, sur des moteurs qui n'ont servi à aucun choix. |
| Vos features utilisent-elles le futur ? | Non. Au cycle *t*, elles n'utilisent que les cycles déjà observés du même moteur. Une vérification dans le notebook et des tests automatiques le confirment. |
| Comment calculez-vous la pente ? | Pour chaque cycle, je prends les 30 dernières mesures du capteur et `np.polyfit` trouve la droite la plus proche de ces points ; son coefficient directeur est la pente, en unités par cycle. |
| Et au début de vie, quand il y a moins de 30 cycles ? | On utilise les cycles disponibles : la moyenne et la pente se calculent sur moins de points (pente 0 au premier cycle), et la référence de `delta` est la moyenne des cycles déjà vus jusqu'au cycle 20. |
| Pourquoi `max_features` = 0,33 ? | À chaque question, chaque arbre ne regarde qu'un tiers des features : les arbres sont plus différents et leur moyenne est plus fiable. J'ai vérifié par validation croisée que 0,33 fait mieux que 1,0 (12,85 contre 13,16). |
| Pourquoi pas de normalisation ? | Elle ne change pas les prédictions d'une régression linéaire ni d'une forêt aléatoire. Elle sera nécessaire pour le LSTM. |
| Le test a-t-il influencé vos choix ? | Pas directement : tous les choix suivent la validation croisée. Mais le test avait été lu par des versions précédentes, ce qui a pu orienter le travail. Je présente donc 14,4 comme un score probablement un peu optimiste, et je m'appuie aussi sur la validation croisée et les 20 moteurs de validation. |
| Pourquoi le test est-il moins bon que la validation ? | 28 moteurs de test sont encore sains (RUL > 110, hors de portée du plafond), 7 ont peu d'historique, et le test ne compte qu'un point par moteur, avec des RUL jusqu'à 145. |
| Que signifie une prédiction trop optimiste ? | Le modèle annonce plus de vie qu'il n'en reste : c'est l'erreur la plus dangereuse. Il faudra en tenir compte dans les seuils d'alerte. |

### Formulations à éviter

| À éviter | À dire plutôt |
|---|---|
| « J'ai choisi le modèle grâce au test » | « J'ai choisi par validation croisée ; le test sert à l'évaluation finale » |
| « Le score test est totalement indépendant » | « Le test avait déjà été consulté : 14,4 est probablement un peu optimiste » |
| « `sensor_3` cause la panne » | « La moyenne de `sensor_3` est la feature la plus utilisée par le modèle » |
| « Le LSTM sera forcément meilleur » | « Le LSTM devra battre la baseline sur le même protocole ; sinon, c'est un résultat à analyser » |
