# Reprendre le projet SentinelX AI

Ce fichier s'adresse à toute personne (ou assistant IA) qui reprend le projet sans historique. Le contexte technique, les résultats et les règles sont dans le [README](../README.md) ; ce fichier ajoute seulement la manière de travailler et un prompt de reprise.

## Profil et préférences de la personne qui mène le projet

- **Premier grand projet en IA** : elle veut **comprendre chaque étape** et pouvoir **défendre ses choix seule** devant le jury, pas seulement obtenir du code.
- **Langue des explications** : **derja tunisienne** (écrite en caractères latins), en gardant les **termes techniques en français ou en anglais** et en expliquant leur sens, avec des exemples simples.
- **Langue des livrables** (notebooks, guides, rapport, README) : **français**, avec accents, ton académique simple et naturel.
- **Méthode attendue** : expliquer progressivement, montrer le pourquoi de chaque changement, ne jamais inventer de résultat, exécuter le code et lire les sorties réelles avant d'écrire une interprétation.
- Garder le **dossier propre** (pas de fichiers temporaires dans le projet) et ne pas modifier l'environnement `.venv` sans le dire.

## Fichiers à lire en premier

1. [`README.md`](../README.md) : contexte, résultats, **règles méthodologiques** (section 7), feuille de route.
2. [`docs/guide_eda.md`](guide_eda.md) et [`docs/guide_baseline.md`](guide_baseline.md) : vue d'ensemble des deux étapes terminées.
3. [`notebooks/01_eda_fd001.ipynb`](../notebooks/01_eda_fd001.ipynb) (sections 12 et 13) et [`notebooks/02_baseline_fd001.ipynb`](../notebooks/02_baseline_fd001.ipynb) (sections 4, 8 et 10), puis `ml/src/` (code testé par `python -m pytest`).
4. `artifacts/eda_fd001/*.json` et `artifacts/baseline_fd001/baseline_metrics.json`.
5. [`docs/cahier_des_charges_sentinelx_ai.pdf`](cahier_des_charges_sentinelx_ai.pdf) pour les exigences et le planning.

## Prompt de reprise (à copier-coller)

```text
Je travaille sur le projet SentinelX AI (maintenance prédictive sur NASA C-MAPSS FD001).
Lis d'abord README.md, puis docs/guide_eda.md et docs/guide_baseline.md : ils contiennent
le contexte, les résultats, les règles anti-leakage et le plan de la prochaine étape.

Où j'en suis : l'EDA (notebooks/01_eda_fd001.ipynb) et la baseline
(notebooks/02_baseline_fd001.ipynb, code dans ml/src/) sont terminées et vérifiées.
Prochaine étape : S4 — LSTM / GRU, à comparer à la baseline (README, section 11).

Comment travailler avec moi :
- C'est mon premier grand projet en IA : explique-moi chaque étape en derja tunisienne,
  avec les termes techniques en français/anglais et leur signification.
- Les notebooks, les guides et le rapport restent en français académique simple.
- N'invente jamais de résultat : exécute le code et base-toi sur les sorties réelles.
- Respecte les règles méthodologiques de la section 7 du README : choix par validation croisée
  sur les 80 moteurs, 20 moteurs de validation utilisés une fois, test lu une seule fois à la fin.
- Avance étape par étape et explique pourquoi chaque choix est fait.
```
