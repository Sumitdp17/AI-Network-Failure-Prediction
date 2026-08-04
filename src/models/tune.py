from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


class HyperparameterTuner:

    def __init__(self):

        self.model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )

        self.best_model = None
        self.best_params = None
        self.search = None
    
    def get_param_grid(self):

        return {

            "n_estimators": [100, 200, 300, 500],

            "max_depth": [
                None,
                10,
                20,
                30,
                40
            ],

            "min_samples_split": [
                2,
                5,
                10
            ],

            "min_samples_leaf": [
                1,
                2,
                4
            ],

            "bootstrap": [
                True,
                False
            ]
        }
    
    def tune(self, X_train, y_train):

        self.search = RandomizedSearchCV(

            estimator=self.model,

            param_distributions=self.get_param_grid(),

            n_iter=20,

            cv=3,

            verbose=2,

            random_state=42,

            n_jobs=-1,

            scoring="f1"
        )

        self.search.fit(
            X_train,
            y_train
        )

        self.best_model = self.search.best_estimator_

        self.best_params = self.search.best_params_

        return self.best_model