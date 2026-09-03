# ⚽ TransferLens

<p align="center">
  <strong>AI-Driven Football Transfer & Player Valuation Analytics</strong>
</p>

<p align="center">
  Machine learning • Football Analytics • Feature Engineering • Predictive Modeling
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/CatBoost-ML-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Football-Analytics-green?style=for-the-badge" />
</p>

---

## 🧠 What is TransferLens?

**TransferLens** is an AI-driven football analytics system built to explore and predict player transfer activity and market value using machine learning.

Instead of looking at football transfers as isolated events, TransferLens combines multiple dimensions of football data to identify patterns that can help understand **why a player may become a transfer candidate and how different player characteristics relate to market value.**

The project brings together:

> **Football Data → Feature Engineering → Machine Learning → Prediction → Analysis**

---

# 🎯 What Does It Do?

TransferLens currently focuses on **two major prediction problems**:

### 🔄 Transfer Prediction

Predicts whether a player is likely to be involved in a transfer using engineered football features.

### 💰 Market Value Prediction

Uses player and football-related characteristics to estimate a player's market value.

Together, these models form the foundation of a football analytics system focused on the **transfer market**.

---

# 🏆 Transfer Prediction Model

The current successful transfer-prediction experiment uses:

| Component | Details |
|---|---|
| 🤖 Model | **CatBoost** |
| 🧩 Features | **72 engineered features** |
| 🎚️ Decision Threshold | **0.51** |
| 🎯 Test Accuracy | **86.02%** |
| 📈 ROC-AUC | **0.9296** |
| ⚖️ F1 Score | **0.7191** |

### 📊 Model Performance

```text
Test Accuracy     █████████████████░░░  86.02%
ROC-AUC           ██████████████████░░  0.9296
F1 Score          ██████████████░░░░░░  0.7191
