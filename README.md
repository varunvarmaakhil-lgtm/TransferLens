# ⚽ TransferLens

<h3 align="center">
AI-Driven Football Transfer & Player Market Analytics
</h3>

<p align="center">
Machine Learning • Football Analytics • Feature Engineering • Predictive Modeling
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-ML-orange?style=for-the-badge)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Football Analytics](https://img.shields.io/badge/Football-Analytics-2E7D32?style=for-the-badge)

</p>

---

## 🧠 What is TransferLens?

**TransferLens** is an AI-driven football analytics system designed to analyze player information, transfer activity and market value using machine learning.

The project combines football data from multiple dimensions and transforms it into structured features that can be used for predictive modeling.

Rather than treating a football transfer as an isolated event, TransferLens attempts to identify relationships between:

- 👤 Player profiles
- 📈 Player performance
- 💰 Market values
- 🔄 Transfer histories
- 🏥 Injury histories
- 🌍 National-team performance
- 🏟️ Team information
- 🏆 Competition and season information
- 🔗 Player/team relationships

The overall idea is to connect **football data → feature engineering → machine learning → prediction → analysis**.

---

# 🎯 Project Objectives

TransferLens currently focuses on two major machine-learning problems.

### 🔄 1. Transfer Prediction

Predict whether a football player is likely to be involved in a transfer using engineered football-related features.

The model learns patterns from player, performance, market and transfer-related information.

---

### 💰 2. Market Value Prediction

Estimate a player's market value using player and football-related characteristics.

This creates a second analytical layer that can complement transfer prediction and provide additional insight into the football transfer market.

---

# 🏗️ Project Architecture

TransferLens follows a complete machine-learning pipeline:

```text
                    ┌─────────────────────┐
                    │   Football Data     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Collection   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Cleaning &     │
                    │ Validation          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Dataset Preparation │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Model Training    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Tuning /      │
                    │ Comparison          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Model Evaluation   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Prediction      │
                    └─────────────────────┘
