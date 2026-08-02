# Phase 1: Exploratory Data Analysis (EDA)

## Objective

The objective of this phase was to understand the structure, quality, and characteristics of the UNSW-NB15 dataset before model development.

---

## Dataset Summary

| Property | Value |
|----------|------:|
| Training Samples | 175,341 |
| Testing Samples | 82,332 |
| Features | 45 |
| Numerical Features | 41 |
| Categorical Features | 4 |
| Missing Values | 0 |
| Duplicate Rows | 0 |

---

## Target Variable

**Column:** `label`

- 0 → Normal Network Traffic
- 1 → Abnormal Network Traffic

### Distribution

| Label | Count | Percentage |
|------:|------:|-----------:|
| Normal (0) | 56,000 | 31.9% |
| Attack (1) | 119,341 | 68.1% |

The dataset is moderately imbalanced. Therefore, model evaluation will include Precision, Recall, F1-score, and ROC-AUC in addition to Accuracy.

---

## Attack Categories

- Normal
- Generic
- Exploits
- Fuzzers
- DoS
- Reconnaissance
- Analysis
- Backdoor
- Shellcode
- Worms

Although attack categories are available, the initial model will focus on binary classification using the `label` column.

---

## Data Quality Assessment

- Missing Values: None
- Duplicate Rows: None
- Constant Columns: None

The dataset is clean and does not require imputation or duplicate removal.

---

## Feature Engineering Decisions

### Features to Remove

| Feature | Reason |
|----------|--------|
| id | Identifier; not a meaningful predictive feature |
| attack_cat | Causes data leakage because it directly reveals the target class |

### Features to Encode

- proto
- service
- state

One-Hot Encoding will be applied during preprocessing.

---

## Correlation Insights

Highly correlated features with the target include:

- sttl
- ct_state_ttl
- rate
- ct_dst_sport_ltm
- ct_src_dport_ltm

These features are expected to contribute significantly to the prediction model.

---

## Key Learnings

- Verified dataset integrity.
- Identified categorical and numerical features.
- Confirmed absence of missing values and duplicates.
- Detected moderate class imbalance.
- Planned preprocessing strategy to prevent data leakage.

---

## Phase Outcome

EDA completed successfully.

The project is ready for data preprocessing and feature engineering.