# COVID-19 Mortality Prediction — End-to-End ML Pipeline

> A complete data science pipeline for predicting patient mortality from COVID-19 using Mexico's national patient registry. Includes exploratory data analysis, six machine learning models with cross-validation, SHAP explainability, and an interactive Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Installation & Setup](#installation--setup)
5. [How to Run](#how-to-run)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Machine Learning Models](#machine-learning-models)
8. [Cross-Validation Results](#cross-validation-results)
9. [SHAP Explainability](#shap-explainability)
10. [Streamlit Dashboard](#streamlit-dashboard)
11. [Key Findings](#key-findings)
12. [Limitations](#limitations)
13. [Tech Stack](#tech-stack)

---

## Project Overview

This project builds a full supervised machine learning pipeline on the **Mexico COVID-19 Open Data Registry** — one of the largest publicly available COVID-19 clinical datasets with over **1 million patient records**.

**Goal:** Predict whether a COVID-19 patient will die (`DEATH = 1`) based on 16 demographic and clinical features, enabling early risk stratification and prioritized healthcare resource allocation.

All six models are treated as **regression estimators** on the binary (0/1) target, producing continuous risk scores that are evaluated with both regression metrics (RMSE, R², MAE) and classification metrics (AUC-ROC). This is a common and valid approach in clinical prediction modelling.

---

## Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | Mexico DGIS — Dirección General de Epidemiología |
| **Total Records** | 1,021,977 patients |
| **Training Sample** | 200,000 (random, `seed=42`) |
| **Features** | 16 clinical & demographic |
| **Target Variable** | `DEATH` (0 = survived, 1 = died) |
| **Mortality Rate** | 7.31% (1 in ~14 patients) |
| **Class Imbalance** | ~1 : 12.7 |
| **Missing Values** | None |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `SEX` | Binary | 0 = Female, 1 = Male |
| `AGE` | Continuous | Patient age in years |
| `HOSPITALIZED` | Binary | 1 = admitted to hospital |
| `PNEUMONIA` | Binary | 1 = pneumonia diagnosed |
| `PREGNANT` | Binary | 1 = pregnant at time of care |
| `DIABETES` | Binary | 1 = diabetic |
| `COPD` | Binary | 1 = chronic obstructive pulmonary disease |
| `ASTHMA` | Binary | 1 = asthma |
| `IMMUNOSUPPRESSION` | Binary | 1 = immunosuppressed |
| `HYPERTENSION` | Binary | 1 = hypertensive |
| `OTHER_DISEASE` | Binary | 1 = other chronic conditions |
| `CARDIOVASCULAR` | Binary | 1 = cardiovascular disease |
| `OBESITY` | Binary | 1 = obese |
| `RENAL_CHRONIC` | Binary | 1 = chronic kidney disease |
| `TOBACCO` | Binary | 1 = tobacco user |
| `COVID_POSITIVE` | Binary | 1 = confirmed COVID-19 positive |

---

## Repository Structure

```
data-science-pipeline/
│
├── covid.csv                        # Raw dataset (1,021,977 rows)
├── train_models.py                  # Full training pipeline (run once)
├── complete_pipeline.py             # Lightweight: SHAP + plots only (skips retraining)
├── app.py                           # Streamlit dashboard
├── requirements.txt                 # Python dependencies
├── run_dashboard.bat                # Windows one-click launcher
│
├── .streamlit/
│   └── config.toml                  # Streamlit server config (headless, port 8501)
│
└── pipeline_outputs/
    ├── cv_results.csv               # 5-fold CV scores for all 6 models
    ├── dataset_stats.json           # Dataset summary metadata
    ├── plot_meta.json               # EDA plot index
    │
    ├── models/                      # Serialized trained models (.pkl)
    │   ├── Linear_Regression.pkl
    │   ├── Lasso.pkl
    │   ├── Ridge.pkl
    │   ├── CART.pkl
    │   ├── Random_Forest.pkl
    │   └── LightGBM.pkl
    │
    ├── plots/                       # EDA & comparison visualizations
    │   ├── 01_class_balance.png
    │   ├── 02_age_distribution.png
    │   ├── 03_age_group_death.png
    │   ├── 04_feature_prevalence.png
    │   ├── 05_death_rate_by_condition.png
    │   ├── 06_correlation_heatmap.png
    │   ├── 07_death_rate_grid.png
    │   ├── 08_sex_analysis.png
    │   ├── 09_age_by_condition.png
    │   ├── 10_comorbidity_cooccurrence.png
    │   ├── 11_hospitalized_sex_death.png
    │   ├── 12_covid_positive_death.png
    │   └── 13_model_comparison.png
    │
    └── shap/                        # SHAP explainability outputs
        ├── shap_values.pkl          # Serialized SHAP arrays (all 6 models)
        ├── shap_beeswarm_*.png      # Global beeswarm plots (6 files)
        ├── shap_bar_*.png           # Feature importance bar charts (6 files)
        ├── shap_dependence_*.png    # Top-3 dependence plots (6 files)
        ├── shap_importance_all_models.png
        └── shap_heatmap_all_models.png
```

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- pip
- Git

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Packages installed:**

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Static visualizations |
| `plotly` | Interactive charts |
| `scikit-learn` | Linear Regression, Lasso, Ridge, CART, Random Forest, CV |
| `lightgbm` | Gradient boosting model |
| `shap` | Model explainability (TreeExplainer, LinearExplainer) |
| `streamlit` | Interactive dashboard |
| `joblib` | Model serialization |
| `scipy` | Statistical utilities |

---

## How to Run

### Option 1 — One-click (Windows)

Double-click `run_dashboard.bat`. This will train models then launch the app automatically.

### Option 2 — Step by step

**Step 1:** Train all models and generate all outputs (run once, takes ~5–10 min):

```bash
python train_models.py
```

**Step 2:** Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

### Option 3 — Skip retraining (models already saved)

If models are already in `pipeline_outputs/models/`, regenerate only SHAP + comparison plots:

```bash
python complete_pipeline.py
streamlit run app.py
```

---

## Exploratory Data Analysis

The pipeline automatically generates **12 EDA plots** covering:

| Plot | Description |
|------|-------------|
| Class Balance | Bar + pie chart showing 7.31% mortality rate |
| Age Distribution | Histogram split by survival outcome; mean age ~44 |
| Age Group Death Rate | Mortality by 5-year bands (dual axis: rate + count) |
| Feature Prevalence | How common each condition is in the dataset |
| Death Rate by Condition | Mortality rate for patients WITH each comorbidity |
| Correlation Heatmap | Full 17×17 Pearson correlation matrix |
| Death Rate Grid | 4×4 grid comparing ON vs OFF mortality per feature |
| Sex Analysis | Death rate and count by sex |
| Age by Condition | Boxplots of age distribution for top 5 comorbidities |
| Comorbidity Co-occurrence | Correlation between conditions |
| Hospitalization × Sex | Interaction between hospitalization status and sex |
| COVID Status vs Death | Mortality for COVID+ vs COVID− patients |

Additionally, an **interactive Plotly chart** in the dashboard shows death rate and patient count across 5-year age bands.

---

## Machine Learning Models

All models are trained as regressors on the binary `DEATH` (0/1) target using a **200,000-patient random sample** and evaluated with **5-fold KFold cross-validation** (`seed=42`).

| Model | Library | Key Hyperparameters |
|-------|---------|-------------------|
| **Linear Regression** | scikit-learn | Default (OLS) |
| **Lasso** | scikit-learn | `alpha=0.0005`, `max_iter=5000` |
| **Ridge** | scikit-learn | `alpha=1.0` |
| **CART** | scikit-learn | `max_depth=8`, `min_samples_leaf=50` |
| **Random Forest** | scikit-learn | `n_estimators=150`, `max_depth=12`, `min_samples_leaf=50`, `n_jobs=-1` |
| **LightGBM** | lightgbm | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`, `num_leaves=63`, `n_jobs=-1` |

**Why regression on a binary target?**  
Treating `DEATH` as a continuous variable (0/1) is a valid approach in clinical prediction modelling. The regression output can be interpreted as a **risk score** or **probability estimate**. AUC-ROC is computed by treating these scores directly as predicted probabilities without thresholding.

---

## Cross-Validation Results

5-fold KFold cross-validation on 200,000 training samples. Green = best in column.

| Model | CV RMSE | CV R² | CV MAE | AUC | Train RMSE | Train R² |
|-------|---------|-------|--------|-----|------------|----------|
| Linear Regression | 0.2123 | 0.3288 | 0.1133 | 0.9495 | 0.2123 | 0.3291 |
| Lasso | 0.2124 | 0.3283 | 0.1129 | 0.9494 | 0.2124 | 0.3285 |
| Ridge | 0.2123 | 0.3288 | 0.1133 | 0.9495 | 0.2123 | 0.3291 |
| CART | 0.2039 | 0.3808 | 0.0823 | 0.9552 | 0.2018 | 0.3940 |
| Random Forest | 0.2034 | 0.3838 | 0.0822 | 0.9583 | 0.2003 | 0.4027 |
| **LightGBM** ✅ | **0.2031** | **0.3857** | **0.0823** | **0.9585** | 0.1987 | 0.4125 |

**Metrics explained:**
- **CV RMSE** — Root Mean Squared Error on held-out fold (lower = better)
- **CV R²** — Coefficient of determination on held-out fold (higher = better)
- **CV MAE** — Mean Absolute Error on held-out fold (lower = better)
- **AUC** — Area Under ROC Curve, computed on full training predictions clipped to [0,1] (higher = better)
- **Train RMSE / Train R²** — Training set performance; compare with CV to assess overfitting

**Takeaway:** All models achieve remarkably high AUC (>0.94), indicating strong discriminative power. LightGBM leads across all metrics. Linear models achieve near-identical performance to Ridge/Lasso, suggesting limited non-linearity at the global level — but tree-based models capture local interaction effects that improve calibration (R² +5.7 pp vs linear).

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) values are computed on a **1,000-patient background sample** for all 6 models using the appropriate explainer:

- **TreeExplainer** — CART, Random Forest, LightGBM (exact, polynomial-time)
- **LinearExplainer** — Linear Regression, Lasso, Ridge (closed-form)

### Plots Generated Per Model

| Plot Type | Description |
|-----------|-------------|
| **Beeswarm** | Each dot = one patient; position = SHAP value; colour = feature value |
| **Bar (Feature Importance)** | Mean \|SHAP\| across all patients — magnitude without direction |
| **Dependence** | Top-3 features: how SHAP value changes with the feature's raw value |

### Cross-Model Plots

| Plot | Description |
|------|-------------|
| `shap_importance_all_models.png` | Normalized importance bar chart — all 6 models side by side |
| `shap_heatmap_all_models.png` | Heatmap: models (rows) × features (columns), normalized to [0,1] |

### Top Predictive Features (LightGBM)

Based on mean |SHAP| values:

1. **AGE** — By far the strongest predictor; risk rises sharply above 50
2. **PNEUMONIA** — Second most important; patients with pneumonia face 4× higher risk
3. **HOSPITALIZED** — Strong signal; closely linked to severity
4. **RENAL_CHRONIC** — Chronic kidney disease compounds age-related risk
5. **DIABETES** — Consistent risk amplifier across all models
6. **CARDIOVASCULAR** — High importance, especially in older patients
7. **HYPERTENSION** — Moderate importance, correlated with age

### SHAP Consistency Across Models

Models agree strongly on feature rankings (low rank standard deviation for AGE, PNEUMONIA, HOSPITALIZED), confirming these are **robustly important** features regardless of model family.

---

## Streamlit Dashboard

The dashboard is organized into **5 tabs**:

### Tab 1 — Executive Summary
- KPI cards: total patients, mortality rate, best AUC, best R²
- Key clinical insights with supporting evidence
- Full model performance table with best-cell highlighting
- Model recommendation with rationale
- Ethical limitations and caveats

### Tab 2 — Data Overview & EDA
- Dataset preview and descriptive statistics
- All 12 static EDA plots
- Interactive Plotly dual-axis chart: death rate + patient count by 5-year age band
- Interactive grouped bar: death rate WITH vs WITHOUT each condition

### Tab 3 — Model Performance
- Full CV results table (RMSE, R², MAE, AUC, Train metrics)
- 4-panel Plotly comparison chart (RMSE, R², AUC, overfitting)
- High-resolution static comparison chart
- Multi-metric radar chart (normalized, all 6 models)

### Tab 4 — Feature Selection _(interactive)_
- 16 individual feature checkboxes grouped by category
- Configurable CV folds (3 / 5 / 10) and sample size (10K / 25K / 50K)
- **"Run Models" button** retrains all 6 models with selected features on-the-fly
- Results table + delta comparison vs full-feature baseline (colour-coded)
- Quick SHAP bar chart for LightGBM with the selected feature set

### Tab 5 — SHAP Analysis
Five sub-views selectable by radio button:

| Sub-view | Content |
|----------|---------|
| **Beeswarm (Global)** | Pre-computed beeswarm + reading guide |
| **Feature Importance Bar** | Pre-computed bar + ranked table + Plotly interactive bar |
| **Dependence Plots** | Pre-computed top-3 + custom interactive scatter with LOWESS trend |
| **Waterfall (Individual)** | Patient slider → Plotly waterfall → risk card (% mortality, risk level, top drivers) |
| **All Models Comparison** | Cross-model importance bar + heatmap (static & Plotly) + ranking agreement chart |

---

## Key Findings

1. **Age dominates all other features** — it is the single strongest predictor across every model family. Patients aged 80+ face >30% mortality vs <1% for under-20s.

2. **Pneumonia is the most dangerous comorbidity signal** — COVID patients with pneumonia have ~28% mortality vs ~6% without (a 4.7× relative increase).

3. **Hospitalization and pneumonia are collinear** — their SHAP dependence plots show overlapping effects; interpreting them independently requires caution.

4. **Tree-based models outperform linear models** — but the margin is modest (AUC delta ~0.009). Linear models achieve AUC >0.949, suggesting the problem is roughly linearly separable in feature space at a global level.

5. **LightGBM is the recommended model** — best AUC (0.9585), best R² (0.3857), fastest training among ensemble methods, and supports native SHAP integration.

6. **Feature ranking is consistent across model families** — AGE, PNEUMONIA, HOSPITALIZED rank in the top 3 for every model, confirming robustness.

7. **Cardiovascular disease and chronic kidney disease interact multiplicatively with age** — their SHAP dependence plots show much higher values for older patients.

---

## Limitations

| Limitation | Details |
|-----------|---------|
| **Class imbalance** | Only 7.3% of patients died. Models were trained on raw proportions without resampling; minority-class recall may be underoptimized |
| **Regression for binary target** | Linear, Lasso, and Ridge can predict values outside [0,1]; predictions are clipped for AUC computation. Logistic variants would be more principled |
| **AUC on training data** | AUC is reported on full training set predictions, not on CV folds — this is optimistic. Future work should compute CV-fold AUC |
| **External validity** | Dataset is from Mexico's healthcare system; performance may differ in other countries or populations |
| **Feature granularity** | Binary encoding loses severity and duration information present in richer EHR systems |
| **Temporal drift** | The dataset covers a specific period of the pandemic; model performance may degrade on later variants |
| **Not a clinical tool** | This project is for educational and research purposes only and must not replace clinical judgment |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Data | pandas, numpy |
| Visualizations | matplotlib, seaborn, plotly |
| ML Models | scikit-learn, lightgbm |
| Explainability | shap (TreeExplainer, LinearExplainer) |
| Serialization | joblib |
| Dashboard | Streamlit |
| Version Control | Git / GitHub |

---

## Reproducibility

All random operations use `seed=42`. To fully reproduce:

```bash
pip install -r requirements.txt
python train_models.py    # ~5–10 min on a modern CPU
streamlit run app.py
```

Expected CV RMSE (LightGBM): `0.2031 ± 0.0010`  
Expected AUC (LightGBM): `0.9585`

---

*Built as part of the MSIS Data Science curriculum — University of Washington, Winter 2026.*
