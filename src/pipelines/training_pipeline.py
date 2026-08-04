from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator

def run_training_pipeline():

    trainer = ModelTrainer()

    trainer.prepare_data()

    trainer.train()
    
    evaluator = ModelEvaluator(
        trainer.model
    )
    evaluator.evaluate(trainer.X_test, trainer.y_test)

    trainer.save_model()

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    run_training_pipeline()