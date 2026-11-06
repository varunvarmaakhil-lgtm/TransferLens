# TransferLens: transfer model evaluation

## Summary

The transfer model predicts whether a player changes club after the current season. Tested season by season on seasons it never saw in training, it reaches **ROC-AUC 0.730 (sd 0.022)** over 2019-2024, against 0.680 for logistic regression on the same features.

An earlier version reported 86% accuracy and 0.93 ROC-AUC. That figure came from a random train/test split and a drifting label, and is withdrawn.

## What was wrong with the original evaluation

1. **Random split.** 84% of test rows belonged to players who also appeared in training.
2. **Label drift.** The label came from Transfermarkt transfer records, which are far less complete for early seasons: 4,055 transfers recorded in 2012/13 vs 17,379 in 2023/24. The labelled transfer rate climbed from 6.6% (2012) to 51% (2023), so early movers were labelled as stayers.
3. **Threshold tuned on the test set.** Small effect: 86.0% vs 85.75% when tuned on validation.
4. **2025-26 rows built separately**, with no player IDs, a same-season label and name-only matching.

Diagnosis on identical features (2012-2024):

| Split | ROC-AUC |
| --- | --- |
| Random rows | 0.921 |
| Grouped by player | 0.810 |
| By time (test 2024) | 0.603 |

## Fixes

- **New label from 1.9M match appearances.** A player's main club in a season is the club he played the most minutes for. Label = 1 if his main club in season t+1 differs from season t. The positive rate is now 28-36% in every season.
- **Rolling-origin evaluation.** For each test season T: train on seasons up to T-2, early-stop and tune the threshold on T-1, test once on T.
- **Removed time and identity proxies:** season, season_year, birth_year, name.
- **17 new time-safe features** (season t uses only data up to the end of season t; market values only before 1 June): minutes and appearance share, change in minutes share, club tenure, new arrival, squad size, minutes rank in squad, club squad turnover, market value, value change, value vs career peak, value vs squad median, squad median value, value rank in squad, club win rate, number of managers, final league position.

## Results (rolling origin, new label, no identity features)

| Test season | Players | Base rate | ROC-AUC | PR-AUC | F1 | LogReg ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 2019 | 5,075 | 0.279 | 0.686 | 0.434 | 0.499 | 0.668 |
| 2020 | 4,217 | 0.300 | 0.734 | 0.604 | 0.549 | 0.682 |
| 2021 | 4,374 | 0.337 | 0.734 | 0.644 | 0.557 | 0.682 |
| 2022 | 4,685 | 0.334 | 0.738 | 0.636 | 0.566 | 0.686 |
| 2023 | 4,716 | 0.335 | 0.749 | 0.658 | 0.578 | 0.685 |
| 2024 | 5,088 | 0.361 | 0.740 | 0.654 | 0.588 | 0.676 |
| **Mean** | | | **0.730** | 0.605 | 0.556 | 0.680 |

Progression:

| Version | ROC-AUC |
| --- | --- |
| Original: random split, transfer-record label | 0.93 (invalid) |
| Rolling origin, appearance label, original features | 0.642 |
| Rolling origin, appearance label, + 17 features, no identity columns | 0.730 |

Top drivers in the final model: club final league position, club win rate, appearance share, squad size, squad median value, value rank in squad, squad turnover. Players tend to leave struggling clubs and clubs where they do not play.

## Limitations

- About a third of player-seasons have no appearances the next season (retired, injured all season, or moved to a league outside the 41 covered competitions). They are excluded from training and evaluation.
- Loans count as moves. January moves count only if they change the main club.
- Contract length, wages and agent data exist only as a current snapshot, so they cannot be used historically without leaking.
- 2019/20 and 2020/21 are COVID-affected seasons.

## Reproduce

Run from this folder with the project environment:

1. python build_label.py (new label from appearances)
2. python features_v2.py (17 time-safe features)
3. python rolling_eval2.py (season-by-season evaluation, writes results/rolling_eval_v2.csv)
4. python train_final.py (final model, writes models/transfer_model_v2.cbm and .json)

Diagnostics: time_split_eval.py (random vs time split), diag2.py (player-grouped split and transfer-record coverage).
