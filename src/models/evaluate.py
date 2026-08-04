import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

class ModelEvaluator:

    def __init__(self, model):

        self.model = model
        
    def predict(self, X_test):

        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):

        predictions = self.predict(X_test)
        
    
        
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(y_test, predictions)

        recall = recall_score(y_test, predictions)

        f1 = f1_score(y_test, predictions)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

       
    
        print("\nClassification Report\n")

        print(classification_report(
                y_test,
                predictions
            ))
            
        cm = confusion_matrix(
                y_test,
                predictions
            )

        display = ConfusionMatrixDisplay(
                confusion_matrix=cm
            )

        display.plot()

        plt.show()
        
        probabilities = self.model.predict_proba(X_test)[:, 1]
            
        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        print(f"ROC-AUC  : {roc_auc:.4f}")
        
        return predictions