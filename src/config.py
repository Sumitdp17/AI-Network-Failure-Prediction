from pathlib import Path

# ============================
# Project Paths
# ============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# ============================
# Dataset Files
# ============================

TRAIN_DATA = RAW_DATA_DIR / "UNSW_NB15_training-set.csv"

TEST_DATA = RAW_DATA_DIR / "UNSW_NB15_testing-set.csv"

# ============================
# Model Artifacts
# ============================

ENCODER_PATH = SAVED_MODELS_DIR / "encoder.pkl"

MODEL_PATH = SAVED_MODELS_DIR / "random_forest.pkl"

# ============================
# ML Settings
# ============================

TARGET_COLUMN = "label"

RANDOM_STATE = 42

# ============================
# Feature Engineering
# ============================

FEATURE_IMPORTANCE_THRESHOLD = 0.01

SELECTED_FEATURES_PATH = SAVED_MODELS_DIR / "selected_features.pkl"