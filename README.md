# IntelliStay: AI-Powered Churn Retention Intelligence Platform

> **1st Place, Megathon 2025 (40+ teams)**

An end-to-end system that predicts auto-insurance customer churn, explains why each customer is at risk, and recommends what to do about it. It combines gradient-boosted modeling, SHAP explanations, DiCE counterfactuals, a local LLM and a Streamlit dashboard.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-green.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

![IntelliStay dashboard](Dashboard.png)

## Team Red Devils

- Siddarth Gottumukkula
- Shlok Sand
- Shreyas Kasture
- M P Samartha
- Vedant Pahariya

Solution deck: [Solution Presentation.pdf](Solution%20Presentation.pdf)

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Overview

Retention teams usually know *that* a customer may leave, but not *why*, and the default response is a blanket discount. IntelliStay closes that gap in four steps:

1. **Predict** churn probability per customer with XGBoost, trained on SMOTE-balanced data.
2. **Explain** the top risk drivers for each customer and across the portfolio with SHAP.
3. **Prescribe** minimal, constrained premium changes that flip the predicted outcome, using DiCE counterfactuals.
4. **Act** on the result: a local LLM drafts a customer persona, retention actions and a behavioral-economics nudge message, and an interactive simulator shows the effect of a given discount before it is offered.

The whole pipeline runs locally. The LLM is served through LM Studio, so there are no API costs and no customer data leaves the machine.

---

## Results

### Model performance (held-out test set)

| Metric    |  Value |
|-----------|-------:|
| Accuracy  |  0.883 |
| Precision |  0.490 |
| Recall    |  0.348 |
| F1-Score  |  0.407 |
| ROC-AUC   |  0.695 |

Churn is the minority class (roughly 1 in 9 customers), so accuracy alone is not informative; precision, recall and F1 on the churn class are the relevant measures. An Optuna-tuned XGBoost was also evaluated against the baseline, and it did not improve F1 or recall, so the baseline model was kept. See `results/model_comparison.png`.

<p>
  <img src="results/overall_performance_updated.png" alt="Overall model performance" width="48%">
  <img src="results/roc_curve.png" alt="ROC curve" width="48%">
</p>

### Top global churn drivers (SHAP)

1. `days_tenure` (shorter tenure, higher risk)
2. `length_of_residence`
3. `home_market_value`
4. `age_in_years`
5. `cust_orig_month`

<p>
  <img src="results/shap_feature_importance_bar.png" alt="SHAP feature importance" width="48%">
  <img src="results/shap_summary_plot.png" alt="SHAP summary plot" width="48%">
</p>

### Portfolio view in the dashboard

| Metric | Value |
|--------|------:|
| Customers scored | 336,182 |
| Average predicted churn risk | 14.7% |
| High risk (>70%) | 1,472 |
| Medium risk (40-70%) | 31,591 |
| Low risk (<40%) | 303,119 |

Revenue at risk is estimated as high-risk customers multiplied by an assumed average annual premium of $950.

### Example: single-customer retention scenario

Customer 63533 starts at a 91.0% predicted churn risk, driven mainly by low `days_tenure`. A DiCE counterfactual with a 25% premium reduction brings the predicted risk to 32.4%, a 58.6 percentage-point drop. The corresponding SHAP waterfall is in `shap_force_plots/`.

---

## Key Features

### Portfolio overview dashboard
- Portfolio-wide churn risk distribution and global feature importance
- Priority list of the highest-risk customers
- Revenue-at-risk estimate and live LLM status indicator

### Customer deep dive
- Individual churn probability with a SHAP waterfall plot
- Top risk factors with direction and magnitude of impact
- LLM-generated customer persona, retention actions and nudge message
- Rule-based fallback insights when the LLM server is unavailable

### Counterfactual retention strategies (DiCE)
- Three diverse retention scenarios per customer
- Only actionable features are modified (premium-related); demographics, tenure and location are held fixed
- Premium can only decrease, capped at a 50% discount

### Interactive retention simulator
- Premium-adjustment slider with instant churn re-scoring
- Expected-value and retention-cost metrics
- Suggested action plan based on the simulated impact

### Behavioral economics layer
Retention messages draw on six established nudge categories, selected by the customer's primary churn driver:

| Nudge | Principle | Example |
|-------|-----------|---------|
| Loss aversion | People weigh losses more heavily than gains | "You'll lose your accident-free bonus" |
| Social proof | People follow peer behavior | "92% of customers in your area renewed" |
| Reciprocity | People return favors | "Complimentary vehicle check-up" |
| Commitment | People value locked-in decisions | "Lock in your rate for 2 years" |
| Scarcity | Limited-time offers drive action | "Offer expires in 48 hours" |
| Anchoring | Reference points shape perceived value | "You've saved $2,340 over 3 years" |

### Local LLM integration
- Qwen3-4B-Thinking (`qwen/qwen3-4b-thinking-2507`) served by LM Studio
- Synthesizes SHAP and DiCE output into plain-language guidance
- Runs offline; setup steps in [lm_studio_setup.md](lm_studio_setup.md)

---

## System Architecture

```
+-------------------------------------------------------------+
|                        DATA PIPELINE                        |
|     Raw CSV -> Feature Engineering -> Train/Test Split      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                       ML MODELING LAYER                     |
|       Scaling -> SMOTE (train only) -> XGBoost classifier   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     EXPLAINABILITY LAYER                    |
|  +--------------+  +---------------+  +------------------+  |
|  | SHAP         |  | DiCE          |  | LLM (Qwen3-4B)   |  |
|  | why churn?   |  | what to change|  | synthesize       |  |
|  +--------------+  +---------------+  +------------------+  |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     PRESENTATION LAYER                      |
|                     Streamlit dashboard                     |
|  +-----------+  +---------------+  +--------------------+   |
|  | Churn     |  | Customer      |  | Retention          |   |
|  | Dashboard |  | Deep Dive     |  | Simulator          |   |
|  +-----------+  +---------------+  +--------------------+   |
+-------------------------------------------------------------+
```

---

## Methodology

### 1. Data preprocessing and feature engineering
- Dropped identifiers and high-cardinality or leakage-prone columns (IDs, dates, city, county)
- Median imputation for numeric columns, mode imputation for categorical columns
- Label encoding for categoricals; `StandardScaler` fit on the training split only
- Engineered features include `premium_to_income_ratio`, `monthly_premium`, `premium_affordability`, `tenure_years`, `age_group`, `income_bracket` and a composite `customer_quality_score`

### 2. Modeling
- Stratified 80/20 train/test split (`random_state=42`)
- SMOTE applied to the training split only, to address class imbalance without contaminating the test set
- XGBoost classifier, trained on GPU when CUDA is available and on CPU otherwise
- Evaluation on the untouched test set: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve

### 3. SHAP explainability
- Global: mean absolute SHAP ranking and summary plots across the portfolio
- Local: per-customer waterfall plots and top-factor tables, rendered on demand in the dashboard

### 4. DiCE counterfactuals
- **Actionable features:** `curr_ann_amt`, `monthly_premium`, `premium_to_income_ratio`, `premium_affordability`
- **Immutable features:** demographics, tenure, location, income, education, credit and customer history
- **Constraints:** premium may only decrease, to at most a 50% discount; three diverse counterfactuals; desired outcome is retention (`Churn = 0`)

### 5. LLM synthesis
SHAP drivers and global context are formatted into a structured prompt that asks the model for a persona, a short explanation of the risk, concrete retention actions and a nudge recommendation with a message template.

---

## Technology Stack

| Area | Tools |
|------|-------|
| Data and ML | Python, Pandas, NumPy, Scikit-learn, imbalanced-learn (SMOTE), XGBoost |
| Explainability | SHAP, DiCE-ML, Matplotlib, Seaborn |
| LLM | LM Studio (OpenAI-compatible local server), Qwen3-4B-Thinking, Requests |
| Dashboard | Streamlit with custom CSS |
| Tooling | Joblib (artifact serialization), Git |

---

## Installation

### Prerequisites
- Python 3.9 or newer
- Optional: an NVIDIA GPU with CUDA for faster XGBoost training (the pipeline falls back to CPU automatically)
- Optional: [LM Studio](https://lmstudio.ai/) for LLM-generated insights

### Setup

```bash
git clone https://github.com/IamSid44/IntelliStay-Megathon-25.git
cd IntelliStay-Megathon-25

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Data and artifacts

The raw dataset (roughly 336,000 rows, key columns `individual_id` and `Churn`), the engineered CSV and the trained `.pkl` artifacts are too large for the repository. Download them from [this Google Drive folder](https://drive.google.com/drive/folders/16kiTYnvHENG4nyMVJsIt0IXIahScT62D?usp=sharing) and place them in the project root.

### LLM setup (optional)

Follow [lm_studio_setup.md](lm_studio_setup.md) to load `qwen/qwen3-4b-thinking-2507` and start the local server on `localhost:1234`. Without it, the dashboard uses rule-based fallback insights.

---

## Usage

### Full pipeline

```bash
python 01_eda_and_preprocessing.py          # EDA, cleaning, feature engineering
python 02_modeling_pipeline.py              # scaling, SMOTE, XGBoost training, evaluation
python 03_shap_explainability_with_llm.py   # SHAP analysis and LLM insights
python dice_counterfactuals.py              # DiCE setup (optional)
streamlit run 04_dashboard.py               # launch the dashboard
```

### Quick start with pre-trained artifacts

Place the downloaded files in the project root and run `streamlit run 04_dashboard.py`. Required files:

- `final_xgboost_model.pkl`
- `scaler.pkl`
- `feature_names.pkl`
- `label_encoders.pkl`
- `shap_data_package.pkl`
- `shap_explainer.pkl`
- `test_data_with_predictions.csv`
- `shap_global_importance.csv`

### Dashboard navigation

1. **Churn Dashboard**: portfolio metrics, risk distribution, global drivers, highest-risk customers.
2. **Customer Deep Dive**: choose a customer by ID or risk level, then run the analysis to see the SHAP explanation, LLM insights and DiCE strategies.
3. **Retention Simulator**: select a customer, adjust the annual premium, and run the simulation to see the predicted impact.

---

## Project Structure

```
IntelliStay-Megathon-25/
|-- 01_eda_and_preprocessing.py           # Cleaning and feature engineering
|-- 02_modeling_pipeline.py               # SMOTE, XGBoost training, evaluation
|-- 03_shap_explainability_with_llm.py    # SHAP analysis and LLM insights
|-- dice_counterfactuals.py               # DiCE counterfactual generation
|-- 04_dashboard.py                       # Streamlit dashboard
|-- llm_utils.py                          # LM Studio client and SHAP plot helpers
|-- lm_studio_setup.md                    # Local LLM setup guide
|-- requirements.txt
|-- Solution Presentation.pdf             # Hackathon solution deck
|-- Dashboard.png                         # Dashboard screenshot
|-- original_data_visualisation/          # EDA plots, confusion matrix
|-- results/                              # Model and SHAP result plots
`-- shap_force_plots/                     # Per-customer SHAP waterfalls
```

Datasets and `.pkl` artifacts are generated by the pipeline or downloaded (see [Installation](#installation)) and are git-ignored.

---

## Limitations and Future Work

Known limitations:

- **Modest discrimination.** Recall of 0.35 and ROC-AUC of 0.695 mean the model is a prioritization aid, not a precise predictor.
- **Target-derived feature.** `state_churn_rate` is computed from the full dataset before the train/test split, so test-set metrics may be optimistic. Recomputing it from training data only is the first planned fix.
- **Uncalibrated probabilities.** Training on SMOTE-resampled data shifts predicted probabilities upward; risk thresholds (40% and 70%) are heuristic until the model is calibrated.
- **Simplified business metrics.** Revenue at risk uses a flat assumed average premium rather than per-customer premiums.

Planned work:

- Fix the `state_churn_rate` leakage and add probability calibration
- Time-based validation and cost-sensitive threshold selection
- A/B testing of nudge messaging against discount-only offers
- Serving the model behind an API and logging outcomes for feedback

---

*From prediction to action: AI that retains customers.*
