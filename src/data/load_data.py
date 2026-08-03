import pandas as pd


def load_data(train_path: str, test_path: str):
    """
    Load training and testing datasets.
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df