# TransferLens

<h3 align="center">
AI-Driven Football Transfer & Player Market Analytics
</h3>

<p align="center">
Machine Learning • Football Analytics • Feature Engineering • Predictive Modeling
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/CatBoost-ML-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Football-Analytics-2E7D32?style=for-the-badge" />
</p>

---

## Overview

**TransferLens** is an AI-driven football analytics system designed to analyze player information, transfer activity, and market value using machine learning.

The system combines football data from multiple dimensions and transforms it into structured features for predictive modeling. Rather than treating a football transfer as an isolated event, TransferLens attempts to identify relationships between player characteristics, performance, market value, transfer history, team information, and other football-related attributes.

The project combines:

> **Football Data → Data Processing → Feature Engineering → Machine Learning → Prediction → Analysis**

TransferLens currently focuses on two primary machine-learning tasks:

1. **Transfer Prediction**
2. **Player Market Value Prediction**

---

# Project Objectives

The primary objective of TransferLens is to build a data-driven football analytics system capable of extracting useful predictive information from large-scale football datasets.

The project focuses on:

- Understanding player characteristics and historical performance
- Transforming raw football information into machine-learning features
- Predicting the likelihood of player transfer activity
- Estimating player market value
- Evaluating machine-learning models using appropriate classification metrics
- Building a usable prediction interface
- Creating a reusable pipeline for football analytics experiments

---

# System Architecture

TransferLens follows a complete machine-learning workflow from raw football data to predictions.

```text
                    ┌─────────────────────────┐
                    │     Football Data        │
                    │ Player / Team / Transfer │
                    │ Performance / Market     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Data Collection       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Cleaning & Validation     │
                    │ Missing Values            │
                    │ Data Consistency          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Feature Engineering       │
                    │ 72 Engineered Features    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Dataset Preparation       │
                    │ Training / Testing        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Model Training            │
                    │ CatBoost / ML Models      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Model Tuning & Comparison │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Model Evaluation          │
                    │ Accuracy / ROC-AUC / F1  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Prediction Pipeline       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Streamlit Application     │
                    │ Player Analysis Interface │
                    └─────────────────────────┘
```

---

# Data Architecture

TransferLens uses football data across multiple dimensions rather than relying on a single player statistic.

The data model incorporates information related to:

| Data Dimension | Purpose |
|---|---|
| Player Profiles | Player identity and demographic information |
| Player Performance | Season-level football performance |
| Market Values | Historical player valuation |
| Transfer History | Historical transfer activity |
| Injury History | Historical injury-related information |
| National Team Performance | International football information |
| Team Information | Player-team relationships and team context |
| Competition Data | League and competition context |
| Season Information | Temporal football context |
| Player/Team Relationships | Relationships between players and clubs |

This multi-dimensional structure allows the machine-learning pipeline to consider several aspects of a player's football profile simultaneously.

---

# Data Processing Pipeline

Raw football data is not directly used by the prediction models.

The data passes through multiple stages before reaching the machine-learning pipeline.

## 1. Data Collection

Football information is collected and organized into structured datasets.

The project contains scripts for collecting and preparing player, transfer, and football-related data.

Key scripts include:

```text
collect_data.py
collect_transfers.py
```

---

## 2. Data Cleaning

The collected data is processed to improve consistency and machine-learning usability.

The cleaning stage deals with issues such as:

- Missing values
- Inconsistent records
- Data formatting
- Duplicate information
- Numerical conversion
- Player and team relationships
- Season-based organization

The objective is to produce a consistent dataset suitable for downstream feature engineering.

---

## 3. Dataset Construction

The cleaned football information is transformed into datasets specifically designed for machine-learning tasks.

Important dataset-building scripts include:

```text
build_training_data.py
prepare_ml_dataset.py
build_transfer_dataset.py
```

The transfer dataset is constructed around the prediction problem of determining whether a player is likely to be involved in transfer activity.

---

# Feature Engineering

Feature engineering is one of the central components of TransferLens.

Instead of feeding raw football records directly into a model, the project transforms available information into structured numerical and categorical features.

The current successful transfer-prediction experiment uses:

> **72 engineered features**

These features represent multiple dimensions of player and football context.

## Feature Categories

### Player Features

Information describing the player and their profile.

Examples of the information represented include:

- Player identity
- Age-related information
- Position
- Player characteristics
- Historical football context

### Performance Features

Features derived from football performance and season-level statistics.

These allow the model to distinguish between different levels of player performance and activity.

### Market Features

Market-related information provides context about a player's valuation and position in the transfer market.

### Transfer Features

Historical transfer information is used to provide context about previous transfer activity.

### Injury Features

Historical injury information can provide additional context about a player's availability and career history.

### Team Features

Team-level information provides context around the player's current football environment.

### Competition and Season Features

Competition and season information helps preserve the temporal and football-contextual structure of the dataset.

### National-Team Features

International football information provides an additional dimension of player performance and experience.

---

# Transfer Prediction

The primary classification task is to predict whether a player is likely to be involved in a transfer.

The model receives engineered player and football-related features and produces a transfer probability.

Conceptually:

```text
Player Data
     │
     ▼
Feature Engineering
     │
     ▼
72 Engineered Features
     │
     ▼
CatBoost Classifier
     │
     ▼
Transfer Probability
     │
     ▼
Decision Threshold = 0.51
     │
     ├───────────────┐
     ▼               ▼
Likely Transfer   Less Likely
```

The classification threshold used by the current successful model is:

> **0.51**

This threshold is used to convert the model's predicted probability into the final transfer classification.

---

# Transfer Prediction Model

The current successful transfer-prediction experiment uses **CatBoost**.

CatBoost was selected as the primary model for the successful experiment because it provides a strong machine-learning approach for structured/tabular data and is suitable for datasets containing a mixture of numerical and categorical information.

## Model Configuration

| Component | Value |
|---|---|
| Model | CatBoost Classifier |
| Feature Count | 72 engineered features |
| Decision Threshold | 0.51 |
| Task | Binary Classification |
| Output | Transfer Probability / Classification |

---

# Model Performance

The current successful transfer-prediction experiment achieved the following results on the test dataset:

| Metric | Result |
|---|---:|
| Test Accuracy | **86.02%** |
| ROC-AUC | **0.9296** |
| F1 Score | **0.7191** |

### Performance Summary

```text
Test Accuracy     █████████████████░░░  86.02%
ROC-AUC           ██████████████████░░  0.9296
F1 Score          ██████████████░░░░░░  0.7191
```

### Metric Interpretation

**Accuracy — 86.02%**

Measures the proportion of test predictions that were classified correctly.

**ROC-AUC — 0.9296**

Measures the model's ability to distinguish between transfer and non-transfer cases across different classification thresholds.

**F1 Score — 0.7191**

Provides a balance between precision and recall and is particularly useful when evaluating classification performance beyond accuracy alone.

---

# Market Value Prediction

TransferLens also contains a separate machine-learning pipeline for estimating player market value.

The market-value model uses player and football-related characteristics to estimate a player's market valuation.

The purpose of this model is to provide a second analytical layer alongside transfer prediction.

```text
Player Information
        │
        ▼
Football Features
        │
        ▼
Feature Processing
        │
        ▼
Market Value Model
        │
        ▼
Estimated Market Value
```

The market-value prediction pipeline is implemented separately from the transfer classification pipeline.

Relevant files include:

```text
train_market_model.py
market_value_model.joblib
market_value_model_info.joblib
```

The application presents the predicted market value alongside the player's football profile and transfer-related analysis.

---

# Prediction Pipeline

The production prediction workflow is separated from the model-training process.

The prediction system loads the trained model and prepares player information for inference.

A simplified representation is:

```text
Player Selection
      │
      ▼
Season Selection
      │
      ▼
Player Data Retrieval
      │
      ▼
Feature Preparation
      │
      ▼
Transfer Model
      │
      ▼
Transfer Probability
      │
      ▼
Transfer Classification
```

The market-value workflow operates independently:

```text
Player Selection
      │
      ▼
Player Features
      │
      ▼
Market Value Model
      │
      ▼
Estimated Market Value
```

The prediction logic is implemented through scripts including:

```text
predict.py
predict_transfer.py
predict_backup.py
```

---

# Model Artifacts

The trained machine-learning models are stored as serialized artifacts.

## Transfer Model

```text
transfer_model.joblib
transfer_model.pkl
transfer_model_info.joblib
```

The transfer model artifact contains the trained classification model used by the application.

## Market Value Model

```text
market_value_model.joblib
market_value_model_info.joblib
```

The model information files store supporting information required by the prediction pipeline.

---

# Application Layer

TransferLens includes a Streamlit-based application that provides an interactive interface for football analysis.

The application allows users to work with player data and model predictions through a graphical interface rather than directly interacting with the underlying Python scripts.

The application includes functionality for:

- Player search
- Season selection
- Player overview
- Historical season information
- Year-by-year statistics
- Career history
- Complete dataset information
- Transfer prediction
- Market-value prediction
- Model-based analysis

The application acts as the interface between the trained machine-learning models and the end user.

---

# Streamlit Application Architecture

```text
                     ┌─────────────────────┐
                     │    Streamlit UI      │
                     └──────────┬──────────┘
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
      Player Search       Season Selection    Player Data
             │                  │                  │
             └──────────────────┼──────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Feature Preparation │
                     └──────────┬──────────┘
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           ▼
        ┌──────────────────┐        ┌──────────────────┐
        │ Transfer Model   │        │ Market Value     │
        │ CatBoost         │        │ Prediction Model │
        └────────┬─────────┘        └────────┬─────────┘
                 │                           │
                 ▼                           ▼
        Transfer Prediction         Estimated Market Value
```

---

# Project Structure

```text
TransferLens/
│
├── data/
│   └── transfer_training_updated.csv
│
├── models/
│
├── project_review/
│   └── scripts/
│       ├── add_squad_numbers.py
│       ├── audit_ml_dataset.py
│       ├── build_transfer_dataset.py
│       ├── collect_data.py
│       ├── collect_transfers.py
│       ├── compare_models.py
│       ├── compare_models_clean.py
│       ├── evaluate_best_model.py
│       ├── predict.py
│       ├── predict_transfer.py
│       ├── prepare_ml_dataset.py
│       ├── train_model.py
│       ├── train_market_model.py
│       └── tune_models.py
│
├── app.py
├── build_training_data.py
├── predict.py
├── train_model.py
├── train_market_model.py
│
├── transfer_model.joblib
├── transfer_model.pkl
├── transfer_model_info.joblib
│
├── market_value_model.joblib
├── market_value_model_info.joblib
│
├── requirements.txt
├── README.md
└── README_data.md
```

---

# Core Scripts

| Script | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `build_training_data.py` | Builds training data |
| `train_model.py` | Trains the transfer prediction model |
| `train_market_model.py` | Trains the market-value model |
| `predict.py` | Prediction workflow |
| `predict_transfer.py` | Transfer prediction workflow |
| `collect_data.py` | Football data collection |
| `collect_transfers.py` | Transfer data collection |
| `prepare_ml_dataset.py` | Prepares data for machine learning |
| `build_transfer_dataset.py` | Builds the transfer dataset |
| `audit_ml_dataset.py` | Audits machine-learning data |
| `compare_models.py` | Compares candidate models |
| `compare_models_clean.py` | Clean model comparison workflow |
| `evaluate_best_model.py` | Evaluates the selected model |
| `tune_models.py` | Model tuning and experimentation |

---

# Technology Stack

| Technology | Role |
|---|---|
| Python | Core programming language |
| Pandas | Data processing and manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine-learning utilities and evaluation |
| CatBoost | Transfer classification model |
| Joblib | Model serialization |
| Streamlit | Interactive application |
| Git | Version control |
| Git LFS | Large model/data file management |

---

# Machine Learning Workflow

The machine-learning process used by TransferLens can be summarized as:

```text
Raw Football Data
       │
       ▼
Data Collection
       │
       ▼
Data Cleaning
       │
       ▼
Data Validation
       │
       ▼
Feature Engineering
       │
       ▼
Training Dataset
       │
       ▼
Model Training
       │
       ▼
Model Tuning
       │
       ▼
Model Comparison
       │
       ▼
Model Evaluation
       │
       ▼
Selected Model
       │
       ▼
Serialized Model Artifact
       │
       ▼
Prediction Application
```

This separation allows the training process and prediction application to remain independent.

---

# Model Development Process

The transfer model was developed through an iterative machine-learning workflow rather than relying on a single training run.

The development process includes:

1. Dataset preparation
2. Feature engineering
3. Training multiple candidate models
4. Model comparison
5. Hyperparameter tuning
6. Evaluation using classification metrics
7. Selection of a successful model
8. Serialization of the trained model
9. Integration with the prediction application

The current successful checkpoint is retained as the primary transfer-prediction model.

---

# Evaluation Strategy

TransferLens does not rely solely on accuracy when evaluating the transfer classifier.

Three primary metrics are reported:

### Accuracy

Measures overall classification correctness.

### ROC-AUC

Measures the model's ability to distinguish between transfer and non-transfer cases across probability thresholds.

### F1 Score

Balances precision and recall and provides additional information about classification performance.

Using multiple metrics provides a more complete view of model performance than accuracy alone.

---

# Technical Design Principles

TransferLens was designed around several principles:

### Separation of Training and Inference

Training scripts are kept separate from prediction and application code.

### Reproducible Model Artifacts

Trained models are serialized so that the application can load an already-trained model without retraining every time.

### Feature-Based Prediction

The prediction models operate on engineered features rather than directly consuming raw football records.

### Modular Pipeline

Data collection, preprocessing, feature engineering, training, evaluation, and prediction are implemented as separate stages.

### Interactive Analysis

The Streamlit application provides an interface for exploring player data and obtaining model predictions.

---

# Data and Model Files

Large datasets and serialized machine-learning models can occupy substantial storage.

The project therefore uses Git LFS for selected large files.

Relevant tracked file types include:

```text
*.pkl
*.joblib
*.csv
```

Large raw data directories that are not required for the main application can be excluded from version control.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/varunvarmaakhil-lgtm/TransferLens.git
cd TransferLens
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If Git LFS is required:

```bash
git lfs install
git lfs pull
```

---

# Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser and provide access to the player analysis and prediction interface.

---

# Training the Models

Transfer prediction model:

```bash
python train_model.py
```

Market-value model:

```bash
python train_market_model.py
```

The resulting model artifacts can then be used by the prediction pipeline and Streamlit application.

---

# Prediction

The prediction scripts can be executed independently of the Streamlit interface.

```bash
python predict.py
```

Transfer-specific prediction:

```bash
python predict_transfer.py
```

---

# Current Transfer Model Results

| Metric | Result |
|---|---:|
| Model | CatBoost Classifier |
| Engineered Features | 72 |
| Decision Threshold | 0.51 |
| Test Accuracy | 86.02% |
| ROC-AUC | 0.9296 |
| F1 Score | 0.7191 |

These results correspond to the current successful transfer-prediction checkpoint used as the project baseline.

---

# Limitations

Transfer prediction is inherently dependent on the quality and availability of historical football data.

Potential limitations include:

- Historical data may contain missing information
- Transfer activity can depend on factors that are difficult to quantify
- Club strategy and managerial decisions are not always directly observable
- Player market values can change rapidly
- Transfer decisions can depend on contextual factors outside the available dataset
- Historical relationships do not necessarily guarantee future transfer behavior

Therefore, TransferLens should be considered a predictive analytics system rather than a system capable of guaranteeing future transfer outcomes.

---

# Future Development

Potential future improvements include:

- Expanding the feature set
- Improving temporal feature engineering
- Incorporating additional football competitions
- Improving transfer-event labeling
- Experimenting with additional machine-learning architectures
- Improving market-value prediction
- Adding model explainability
- Adding feature-importance analysis
- Incorporating more advanced player-performance metrics
- Improving real-time data integration
- Expanding the application with comparative player analysis
- Developing richer transfer-market analytics

---

# Project Status

TransferLens currently contains:

- A structured football data pipeline
- A feature-engineering pipeline
- A trained transfer classification model
- A separate market-value prediction model
- Serialized model artifacts
- Model evaluation and comparison workflows
- A Streamlit-based interactive application
- Player and season-level analysis
- Transfer prediction
- Market-value prediction

The current transfer model provides the project's established baseline with:

> **86.02% Test Accuracy | 0.9296 ROC-AUC | 0.7191 F1**

---

# Repository

**GitHub:**  
https://github.com/varunvarmaakhil-lgtm/TransferLens

---

# Author

**Akhil Varun Varmak Kalidindi**

B.Tech Artificial Intelligence  
Mahindra University, Hyderabad

---

## Summary

TransferLens brings together football data engineering, feature engineering, machine learning, and interactive analytics into a single football transfer analysis system.

The project demonstrates an end-to-end workflow:

```text
Data
  ↓
Processing
  ↓
Feature Engineering
  ↓
Machine Learning
  ↓
Evaluation
  ↓
Prediction
  ↓
Interactive Analytics
```

> **Turning football data into predictive insight.**
