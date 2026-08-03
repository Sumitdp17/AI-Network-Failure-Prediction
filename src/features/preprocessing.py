import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessor:

    def __init__(self):
        self.encoder = None

    def save_encoder(self, path: str):

       path = Path(path)
       path.parent.mkdir(parents=True, exist_ok=True)

       joblib.dump(self.encoder, path)

       print(f"Encoder saved to: {path}")

    def drop_unnecessary_columns(self, df):

        df = df.drop(columns=["id", "attack_cat"])

        return df

    def split_features_target(self, df):

         X = df.drop("label", axis=1)

         y = df["label"]

         return X, y

    def encode_features(self, X_train, X_test):

         categorical_columns = [
            "proto",
            "service",
            "state"
        ]

         self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
         )

         encoded_train = self.encoder.fit_transform(
            X_train[categorical_columns]
        )

         encoded_test = self.encoder.transform(
            X_test[categorical_columns]
        )

         encoded_train_df = pd.DataFrame(
            encoded_train,
            columns=self.encoder.get_feature_names_out(
                categorical_columns
            ),
            index=X_train.index
        )

         encoded_test_df = pd.DataFrame(
            encoded_test,
            columns=self.encoder.get_feature_names_out(
                categorical_columns
            ),
            index=X_test.index
        )

         X_train = X_train.drop(columns=categorical_columns)
         X_test = X_test.drop(columns=categorical_columns)

         X_train = pd.concat(
            [X_train, encoded_train_df],
            axis=1
        )

         X_test = pd.concat(
            [X_test, encoded_test_df],
            axis=1
        )

         return X_train, X_test