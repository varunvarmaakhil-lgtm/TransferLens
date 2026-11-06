# TransferLens

AI-assisted football recruitment and decision support for Europe's Big-5 leagues. Work in progress.

The goal: ask in plain English why a player is likely to move, who fits a club, what a fair price is, and what squad a budget buys. Every number comes from ML models, SQL and an optimiser; a small local language model only chooses tools and explains their results.

## Status

**Phase 1 complete: an honest transfer model.**

- Predicts whether a player changes club after the current season.
- **ROC-AUC 0.730 (sd 0.022)**, tested season by season on 2019-2024, each season unseen in training. Logistic regression on the same features: 0.680.
- An earlier version claimed 86% accuracy (0.93 ROC-AUC). An audit found a random-split leak and a label that drifted with data coverage, so that figure is withdrawn. The full story is in [EVALUATION.md](EVALUATION.md).

| Version | ROC-AUC |
| --- | --- |
| Original: random split, transfer-record label | 0.93 (invalid) |
| Season-by-season, label rebuilt from match appearances | 0.642 |
| + 17 time-safe club and playing-time features | **0.730** |

What drives a move: the club's final league position and win rate, the player's share of appearances, squad size and squad value. Players leave struggling clubs and clubs where they do not play.

## How it works

- **Data:** the open Nodeball Football Datalake (Transfermarkt-based) in DuckDB: 1.9M match appearances, market valuations, games and clubs, 2012-2026.
- **Label:** a player's main club is the club he played the most minutes for; label = 1 if it changes from season t to t+1.
- **Features:** 84 per player-season, all computed from data available at the end of that season.
- **Model:** CatBoost, rolling-origin evaluation (train up to T-2, tune on T-1, test on T).

## Roadmap

1. Honest evaluation and transfer model (done)
2. Fair-price ranges with uncertainty, undervalued-player detection, player similarity search
3. Tool layer with strict schemas plus a local open model (Qwen or Llama via Ollama)
4. SHAP explanations for every prediction
5. Squad and budget optimiser (integer programming with PuLP)
6. Agent evaluation on 50 known-answer questions, Streamlit app, demo

## Repository

- build_label.py, features_v2.py: label and features
- rolling_eval2.py, train_final.py: evaluation and final model
- time_split_eval.py, diag2.py: leakage diagnostics
- results/: every metric quoted above, as CSV
- app.py: current Streamlit app (to be rebuilt in Phase 6)

## Data credit

Data from the Nodeball Football Datalake by salimt (github.com/salimt), built on Transfermarkt. The datalake's own documentation is in README_datalake.md.
