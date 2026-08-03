# Phase 2: Data Preprocessing

## Objective

The objective of this phase was to build a reusable preprocessing pipeline for the UNSW-NB15 dataset. The pipeline prepares raw network traffic data for machine learning while preventing data leakage.

---

## Tasks Completed

- Loaded training and testing datasets.
- Removed unnecessary columns.
- Split features and target variable.
- Encoded categorical features using One-Hot Encoding.
- Saved the fitted encoder for future inference.

---

## Columns Removed

| Column | Reason |
|---------|--------|
| id | Identifier; not useful for prediction |
| attack_cat | Causes data leakage because it directly reveals the target class |

---

## Target Variable

label

- 0 → Normal Traffic
- 1 → Abnormal Traffic

---

## Categorical Features

The following features were encoded:

- proto
- service
- state

Encoding Technique:

- One-Hot Encoding

Configuration:

- handle_unknown="ignore"
- sparse_output=False

---

## Why Fit Only on Training Data?

The encoder was fitted only on the training dataset and then used to transform the testing dataset.

This prevents data leakage and simulates real-world deployment where future data is unseen during training.

---

## Encoder Persistence

The fitted encoder was saved as:

saved_models/encoder.pkl

This encoder will later be reused by the FastAPI prediction service.

---

## Output

Training Features:

175341 × 194

Testing Features:

82332 × 194

The increase in feature count is due to One-Hot Encoding of categorical variables.

---

## Technologies Used

- Pandas
- Scikit-learn
- OneHotEncoder
- Joblib
- Pathlib

---

## Skills Demonstrated

- Modular Python Programming
- Feature Engineering
- Data Preprocessing
- Data Leakage Prevention
- Reusable ML Pipelines
- Model Artifact Serialization

---

## Interview Questions

### Why did you remove attack_cat?

The attack category directly reveals whether a sample is normal or malicious. Including it would cause data leakage because the model could infer the target instead of learning meaningful patterns.

---

### Why One-Hot Encoding instead of Label Encoding?

The categorical variables have no ordinal relationship. One-Hot Encoding avoids introducing artificial ordering between categories.

---

### Why save the encoder?

The same encoder must be used during inference to ensure new data is transformed exactly as it was during training.