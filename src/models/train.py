import joblib

from sklearn.ensemble import RandomForestClassifier

from src.data.load_data import load_data
from src.features.preprocessing import DataPreprocessor

from src.config import (
    TRAIN_DATA,
    TEST_DATA,
    MODEL_PATH,
    RANDOM_STATE,
    ENCODER_PATH,
)

class ModelTrainer:

    def __init__(self):

        self.preprocessor = DataPreprocessor()

        self.model = RandomForestClassifier(
            random_state=RANDOM_STATE
        )

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None
        
    def prepare_data(self):
        """
        Load datasets and perform preprocessing.
        """

        # Load datasets
        train_df, test_df = load_data(
            TRAIN_DATA,
            TEST_DATA
        )

        # Remove unnecessary columns
        train_df = self.preprocessor.drop_unnecessary_columns(train_df)
        test_df = self.preprocessor.drop_unnecessary_columns(test_df)

        # Split features and labels
        X_train, y_train = self.preprocessor.split_features_target(train_df)
        X_test, y_test = self.preprocessor.split_features_target(test_df)

        # Encode categorical features
        X_train, X_test = self.preprocessor.encode_features(
            X_train,
            X_test
        )

        # Store inside the class
        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test

        # Save encoder
        self.preprocessor.save_encoder(
            str(ENCODER_PATH)
        )

        print("Data preprocessing completed successfully.")
        
    
    def train(self):
        """
        Train the Random Forest model.
        """

        self.model.fit(
            self.X_train,
            self.y_train
        )

        print("Random Forest training completed.") 
    
    def save_model(self):
        """
        Save trained model.
        """

        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            self.model,
            MODEL_PATH
        )

        print(f"Model saved to {MODEL_PATH}")           