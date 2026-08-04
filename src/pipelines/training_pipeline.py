from src.models.train import ModelTrainer


def run_training_pipeline():

    trainer = ModelTrainer()

    trainer.prepare_data()

    trainer.train()

    trainer.save_model()

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    run_training_pipeline()