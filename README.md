# Childhood Autism Detection using Machine Learning

## Overview

Every year, thousands of children miss out on critical autism support simply because diagnosis takes too long. Early intervention before age 5 can make a life-changing difference, but clinic backlogs often delay answers by months or even years.

This project uses machine learning to build a fast, reliable pre-screening tool that can flag children who may need further evaluation, based on a 10-question behavioral checklist and basic demographic information. Trained on 2,179 pediatric records, the best model correctly identified potential ASD cases with 99.1% accuracy, helping get the right children in front of specialists sooner.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0746a876-77ab-4cf5-9f22-1a4928e12c85" width="600"/>
</p>

---
## Problem Statement

Clinical diagnosis of ASD is time-intensive, expensive, and requires specialist expertise that is unevenly distributed across regions and socioeconomic groups. Validated behavioral screening tools like the Q-CHAT-10 exist, but their results are still interpreted manually or left unprocessed. The opportunity is to automate the classification step: given a completed Q-CHAT-10 form and basic demographic information, predict whether a child is likely to have ASD, enabling faster triage and earlier referral. The challenge is class imbalance (the dataset skews toward positive cases) and injected label noise (3% of labels were randomly flipped to simulate real-world annotation error), which requires robust preprocessing and model selection.

---

## Dataset

**`data/Data.csv`** — 2,179 rows × 18 columns, pediatric autism screening records based on the Q-CHAT-10 questionnaire.


| Variable | Description |
|---|---|
| `A1`–`A10` | Binary (0/1) responses to the 10 Q-CHAT-10 behavioral screening questions |
| `Age_Years` | Child's age in years (range: 1–9) |
| `Qchat-10-Score` | Total Q-CHAT-10 score (sum of A1–A10, range: 0–10) |
| `Sex` | Child's sex (`f` / `m`) |
| `Ethnicity` | Ethnic background (13 raw categories, normalized to 7) |
| `Jaundice` | Whether the child was born with jaundice (`yes` / `no`) |
| `Family_mem_with_ASD` | Whether a family member has a confirmed ASD diagnosis (`yes` / `no`) |
| `Who completed the test` | Relationship of the respondent to the child (family member, health care professional, etc.) |
| `Class` | **Target variable** — `Yes` (ASD likely) / `No` |

**Class distribution (post noise injection):** 1,335 positive (`Yes`) · 843 negative (`No`)

**Engineered features added during preprocessing:**
- `Score` — renamed from `Qchat-10-Score`
- `Family_History` — binary flag derived from `Who completed the test` (family member = 1)
- `Ethnicity_*` — 6 one-hot encoded columns from normalized ethnicity categories

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Preprocessing & splitting | `scikit-learn` (`train_test_split`, `LabelEncoder`) |
| Class imbalance handling | `imbalanced-learn` (`SMOTE`, `RandomOverSampler`) |
| Classical ML models | `scikit-learn` (`LogisticRegression`, `GaussianNB`, `SVC`, `RandomForestClassifier`) |
| Gradient boosting | `xgboost` (`XGBClassifier`) |
| Deep learning | `tensorflow` / `keras` (Sequential API) |
| Evaluation | `scikit-learn` (`classification_report`, `roc_auc_score`, `matthews_corrcoef`, `StratifiedKFold`, `LeaveOneOut`) |

---

## Project Pipeline

```
Raw Data
   |
   v
1. Data Pre-Processing
   |-- Label noise injection: 3% of Class labels randomly flipped (robustness simulation)
   |-- Binary encoding:
   |     |-- Sex:                  m → 1, f → 0
   |     |-- Jaundice:             yes → 1, no → 0
   |     |-- Family_mem_with_ASD:  yes → 1, no → 0
   |     |-- Class:                Yes → 1, No → 0
   |     +-- Who completed test:   family member → 1, all others → 0 (renamed Family_History)
   |-- Ethnicity normalization: 13 raw categories → 7 clean groups
   |     |-- Hispanic + Latino → Hispanic/Latino
   |     +-- mixed, Pacifica, Native Indian, PaciFica, Mixed → Others
   |-- Missing value check: none found across all columns
   +-- Column drops: raw Jaundice (post-encoding), Who completed the test (post-encoding)
   |
   v
2. Feature Engineering
   |-- Score            = renamed Qchat-10-Score
   |-- Family_History   = binary flag from Who completed the test
   +-- Ethnicity_*      = 6 one-hot encoded columns (drop_first=True)
   Final feature set: 20 features (A1–A10, Age_Years, Score, Sex, Family_History, 6 ethnicity dummies)
   |
   v
3. Exploratory Data Analysis (EDA)
   |-- Class distribution and prior probabilities
   |-- Per-feature value counts and distributions
   +-- Demographic breakdowns by sex, age, ethnicity, and family history
   |
   v
4. Dataset Splitting
   +-- 80/20 stratified train-test split (random_state=42)
       Training: ~1,742 rows | Test: 436 rows (held out, never resampled)
   |
   v
5. Class Imbalance Handling (training set only)
   |-- SMOTE (sampling_strategy='minority') → balances both classes to 1,063 samples each
   +-- RandomOverSampler → expands training set to 2,500 samples (1,251 pos / 1,249 neg)
   |
   v
6. Model Training (6 algorithms)
   |-- Logistic Regression   (lbfgs solver, random_state=42)
   |-- Naive Bayes            (GaussianNB, default hyperparameters)
   |-- SVM                    (RBF kernel, probability=True, random_state=0)
   |-- Random Forest          (n_estimators=100, max_depth=5, oob_score=True)
   |-- XGBoost                (XGBClassifier, binary:logistic objective, defaults)
   +-- Neural Network         (Dense(8,tanh) → Dense(4,tanh) → Dense(1,sigmoid),
                               binary_crossentropy, Adam, 150 epochs, batch=10)
   |
   v
7. Model Evaluation
   |-- Stratified 5-fold cross-validation
   |-- Standard 5-fold cross-validation
   |-- Leave-One-Out (LOO) cross-validation
   |-- Held-out test set metrics: Accuracy, Precision, Recall, F1-Score, MCC
   +-- ROC curves and Precision-Recall curves (all 6 models compared)
   |
   v
8. Inference Demo
   +-- 10 random test-set rows passed through XGBoost; predictions vs. ground truth.
```

---

## Results

### Model Comparison

| Model | Stratified CV | CV Accuracy | LOO Accuracy | Test Accuracy | Precision | Recall | F1-Score | MCC |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9084 | 0.9084 | 0.9076 | 0.8991 | 0.9254 | 0.9118 | 0.9185 | 0.7862 |
| Naive Bayes | 0.8768 | 0.8768 | 0.8796 | 0.8716 | 0.9000 | 0.8934 | 0.8967 | 0.7270 |
| SVM (RBF) | 0.9564 | 0.9564 | 0.9560 | 0.9427 | 0.9732 | 0.9338 | 0.9531 | 0.8807 |
| Random Forest | 0.9584 | 0.9584 | 0.9520 | 0.9450 | 1.0000 | 0.9118 | 0.9538 | 0.8918 |
| XGBoost | 0.9920 | 0.9920 | 0.9920 | 0.9794 | 0.9888 | 0.9779 | 0.9834 | 0.9563 |
| **Neural Network** | — | — | — | **0.9908** | **1.0000** | **0.9853** | **0.9926** | **0.9807** |


### Selected Model: Neural Network

The Keras Sequential model achieved the highest test accuracy (**99.1%**) and a near-perfect Matthews Correlation Coefficient of **0.981**, indicating reliable discrimination even under class imbalance. XGBoost is the recommended alternative where inference speed and interpretability are priorities.

---

## Key Analytical Findings

- The Q-CHAT-10 individual item scores (`A1`–`A10`) and total score are the dominant predictors behavioral signals captured by the questionnaire carry far more discriminative power than demographic features alone.
- SMOTE combined with random oversampling to a 2,500-sample balanced training set substantially improved minority-class recall compared to training on the raw imbalanced data.
- 3% label noise injection did not prevent XGBoost or the Neural Network from achieving near-perfect accuracy, confirming strong signal-to-noise robustness of these models on this dataset.
- Tree-based and deep learning models (Random Forest, XGBoost, Neural Network) significantly outperform linear models (Logistic Regression, Naive Bayes) the decision boundary for ASD classification is non-linear.
- Random Forest achieved perfect precision (1.000) on the test set, meaning every positive prediction it made was correct, at the cost of slightly lower recall compared to the Neural Network.
- Geographic/ethnic features contribute marginally one-hot encoded ethnicity columns add minor signal but the model generalizes across demographic groups.
- Family history of ASD is a meaningful secondary predictor, consistent with the established heritability of ASD.

---

## Limitations and Future Work

The reported accuracy figures particularly the Neural Network's 99.1% should be interpreted with caution. Several factors inflate these numbers and limit how far the current results generalize:

**Dataset size.** 2,179 records is a small sample for a supervised classification task with 20 features. At this scale, models especially non-parametric ones like XGBoost and neural networks can fit the training distribution closely and still appear to perform well on the held-out split, but the held-out split itself is only 436 samples. A model achieving 99.1% on 436 samples means roughly 4 misclassifications; small shifts in the split seed could meaningfully change that number.

**Synthetic oversampling on a small dataset.** SMOTE generates synthetic minority-class samples by interpolating between existing ones. On a dataset of ~2,179 rows, this risks creating synthetic points that are unrealistically close to the decision boundary, inflating cross-validation performance on the resampled training set in ways that do not reflect true generalization.

**No external validation.** All evaluation was performed on a single train-test split from the same source dataset. The model has never been tested on records from a different clinic, country, or time period. Performance on a genuinely out-of-distribution population is unknown.

---

### What would meaningfully improve this project

- **Larger and more diverse dataset** — scaling to 10,000+ records across multiple clinical sites would reduce overfitting risk and test cross-population generalizability.
- **Explainability layer** — integrating SHAP values or LIME explanations would identify which Q-CHAT-10 items and demographic variables drive predictions for individual children, making the model more clinically interpretable and trustworthy.
- **Longitudinal validation** — testing model predictions against follow-up clinical diagnoses rather than against the original questionnaire label would be the gold-standard validation for a screening tool.
- **Calibration** — converting raw predictions to well-calibrated probabilities (via Platt scaling or isotonic regression) would allow the model to communicate a risk score rather than a binary flag, which is more actionable in a clinical triage setting.

---

## Repository Structure

```
childhood-autism-detection-ml/
|-- images              # plots
|-- code.ipynb          # Main notebook (full pipeline: preprocessing → EDA → training → evaluation)
|-- data/
|   +-- Data.csv        # Input dataset (2,179 pediatric screening records)
|-- report.pdf          # report 
+-- README.md
```
