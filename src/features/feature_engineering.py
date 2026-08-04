import joblib
import pandas as pd

from src.config import (
    FEATURE_IMPORTANCE_THRESHOLD,
    SELECTED_FEATURES_PATH,
)


class FeatureEngineer:
    """
    Handles feature importance calculation, feature selection,
    feature transformation, and saving selected features.
    """

    def __init__(self, model):
        self.model = model
        self.feature_importance = None
        self.selected_features = None

    def calculate_feature_importance(self, X_train):
        """
        Calculate feature importance using the trained model.
        """

        self.feature_importance = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": self.model.feature_importances_
        })

        self.feature_importance.sort_values(
            by="Importance",
            ascending=False,
            inplace=True
        )

        self.feature_importance.reset_index(
            drop=True,
            inplace=True
        )

        return self.feature_importance

    def select_features(
        self,
        threshold=FEATURE_IMPORTANCE_THRESHOLD
    ):
        """
        Select features whose importance is greater than the threshold.
        """

        if self.feature_importance is None:
            raise ValueError(
                "Run calculate_feature_importance() before selecting features."
            )

        self.selected_features = self.feature_importance[
            self.feature_importance["Importance"] > threshold
        ]["Feature"].tolist()

        print(f"Selected {len(self.selected_features)} features.")

        return self.selected_features

    def transform(self, X):
        """
        Keep only the selected features.
        """

        if self.selected_features is None:
            raise ValueError(
                "Run select_features() before transforming the data."
            )

        return X[self.selected_features]

    def save_selected_features(self):
        """
        Save selected feature names.
        """

        if self.selected_features is None:
            raise ValueError(
                "No selected features available to save."
            )

        SELECTED_FEATURES_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            self.selected_features,
            SELECTED_FEATURES_PATH
        )

        print(
            f"Selected features saved to: {SELECTED_FEATURES_PATH}"
        )