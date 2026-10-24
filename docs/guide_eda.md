# Guide de l'EDA — NASA C-MAPSS FD001

Résumé du notebook [`notebooks/01_eda_fd001.ipynb`](../notebooks/01_eda_fd001.ipynb), pour réviser et préparer la soutenance. Le notebook reste la référence : il contient le code, les graphiques et les explications détaillées.

## 1. À quoi sert l'EDA ?

L'**analyse exploratoire des données** (EDA) consiste à comprendre les données **avant** de construire un modèle : vérifier qu'elles sont propres, voir comment les capteurs évoluent avant la panne, construire la valeur à prédire (le RUL) et décider quelles variables garder.

## 2. Vocabulaire

| Mot | Signification |
|---|---|
| Cycle | une période de fonctionnement du moteur |
| Trajectoire | toutes les mesures d'un moteur, du premier cycle au dernier |
| RUL | nombre de cycles restant avant la panne : ce que le modèle devra prédire |
| Feature | variable d'entrée d'un modèle |
| Data leakage | fuite d'information : le modèle voit une information qu'il n'aura pas en situation réelle |
| Validation | moteurs mis de côté pour vérifier le modèle sur des données jamais vues |
| Moyenne glissante | moyenne des derniers cycles, recalculée à chaque cycle |
| Corrélation | nombre entre −1 et +1 qui mesure si deux variables évoluent ensemble (Pearson : en ligne droite ; Spearman : toujours dans le même sens) |
| RUL plafonné | `min(RUL, plafond)` : un moteur sain reçoit la valeur « au moins *plafond* cycles » |

## 3. Le notebook étape par étape

| Section | Ce que nous faisons | Ce que nous obtenons |
|---|---|---|
| 2. Chargement | nommer les 26 colonnes, charger train et test | train : 20 631 lignes, test : 13 096 lignes, 100 moteurs chacun |
| 3. Qualité | valeurs manquantes, doublons, cycles continus | aucun problème |
| 4. Durée de vie | dernier cycle de chaque moteur | 128 à 362 cycles (médiane 199) |
| 5. Séparation | mélange des 100 moteurs avec une graine fixe, 20 pour la validation | 80 moteurs d'analyse (16 561 lignes), 20 de validation (4 070) |
| 6. Choix des capteurs | nombre de valeurs différentes par colonne | 7 colonnes constantes ; `sensor_6` presque constant ; **14 capteurs retenus** |
| 7. Cible RUL | RUL = dernier cycle − cycle courant, vérifié | RUL de 0 à 361 ; zone 0 à 127 couverte par tous les moteurs |
| 8. Évolution des capteurs | capteurs (lissés sur 10 cycles) en fonction du RUL | dégradation lente puis accélérée ; chaque moteur part de son propre niveau |
| 9. Lien avec le RUL | corrélations de Pearson et de Spearman | 12 capteurs nettement liés au RUL (Spearman de 0,56 à 0,71 en valeur absolue) |
| 10. Redondance | corrélations entre capteurs | 7 paires au-dessus de 0,80, dont `sensor_9`/`sensor_14` (0,97) ; tous gardés |
| 11. Test | qualité et historique disponible | 31 à 303 cycles par moteur → historique de 30 cycles au maximum |
| 12. Export | règles anti-fuite et fichiers JSON | décisions lues par la baseline |

## 4. Les décisions transmises à la baseline

1. Utiliser les **mêmes 80 / 20 moteurs** (`artifacts/eda_fd001/fd001_engine_split.json`).
2. Utiliser les **14 capteurs retenus** : `sensor_2`, `3`, `4`, `7`, `8`, `9`, `11`, `12`, `13`, `14`, `15`, `17`, `20`, `21`.
3. Donner au modèle l'**historique récent** (moyenne, tendance, écart au début de vie), car les mesures sont bruitées et chaque moteur part d'un niveau différent.
4. Utiliser **30 cycles d'historique au maximum** (le test a des moteurs de 31 cycles).
5. **Plafonner le RUL**, avec une valeur choisie par validation croisée sur les 80 moteurs d'entraînement (dans la baseline).
6. Évaluer sur la **zone RUL ≤ 127**.
7. Respecter les règles anti-fuite : séparation par moteur, passé uniquement, `RUL_FD001.txt` lu une seule fois à la fin.

## 5. Préparation de la soutenance

### Résumé en une minute

> « Avant de construire un modèle, j'ai analysé le dataset NASA C-MAPSS FD001 : 100 moteurs suivis jusqu'à la panne, soit 20 631 lignes. Les données sont propres : pas de valeur manquante, pas de doublon. J'ai mis de côté 20 moteurs pour la validation, en séparant par moteur pour éviter toute fuite d'information. J'ai écarté 7 colonnes constantes, un capteur presque constant et deux réglages sans lien avec la panne, et j'ai gardé 14 capteurs. J'ai construit la cible RUL et vérifié qu'elle est correcte. Les capteurs évoluent lentement puis de plus en plus vite avant la panne, et chaque moteur part de son propre niveau : c'est pourquoi la baseline utilise l'historique récent des capteurs. Enfin, le jeu de test impose un historique de 30 cycles au maximum. »

### Questions probables du jury

| Question | Réponse |
|---|---|
| Pourquoi faire une EDA avant le modèle ? | Pour vérifier la qualité des données, comprendre leur structure, écarter les variables inutiles et repérer les risques de fuite d'information. |
| Pourquoi séparer par moteur et pas par ligne ? | Deux cycles successifs d'un même moteur sont presque identiques. Avec une séparation par ligne, le modèle « reconnaîtrait » le moteur et ses scores seraient trop optimistes. |
| Pourquoi séparer avant l'analyse ? | Les analyses guident nos choix. Si elles utilisaient la validation, celle-ci ne serait plus indépendante. |
| Pourquoi ne pas lire `RUL_FD001.txt` ? | C'est la réponse de l'examen final. La regarder pourrait influencer nos choix. |
| Pourquoi avoir retiré `sensor_6` et les réglages ? | `sensor_6` est presque constant, et les réglages varient très peu dans FD001. Aucun des trois n'a de lien avec le RUL (corrélation au plus 0,12 en valeur absolue). |
| Pourquoi garder `sensor_9` et `sensor_14` malgré leur faible corrélation ? | Une faible corrélation ne prouve pas qu'une variable est inutile. Leur comportement en fin de vie peut aider un modèle non linéaire. |
| Pourquoi Spearman en plus de Pearson ? | La dégradation n'est pas en ligne droite. Spearman mesure un lien qui va toujours dans le même sens, même s'il n'est pas linéaire. |
| Pourquoi ne pas utiliser `cycle` comme entrée ? | Il est lié mécaniquement au RUL. Nous voulons que le modèle apprenne l'état du moteur à partir des capteurs, pas seulement son âge. |
| Pourquoi un historique de 30 cycles ? | Certains moteurs de test n'ont que 31 cycles : avec 30, tous les moteurs ont assez d'historique. |
| Quelles sont les limites ? | Données simulées, une seule condition de fonctionnement, peu de moteurs (80 pour les choix, 20 pour la vérification). |

### Formulations à éviter

| À éviter | À dire plutôt |
|---|---|
| « `sensor_11` cause la panne » | « `sensor_11` est lié à la baisse du RUL » |
| « J'ai 20 631 moteurs » | « J'ai 100 moteurs et 20 631 lignes moteur-cycle » |
| « `sensor_11` est une température » | « Les capteurs ne sont pas nommés : l'analyse est statistique » |
| « La validation, c'est le test » | « La validation sert à choisir ; le test sert uniquement à l'évaluation finale » |
