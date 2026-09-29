import logging
import os

import joblib
import pandas as pd
import yaml

from sklearn.feature_extraction.text import CountVectorizer


logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    "feature_engineering_errors.log"
)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logger.debug(
            "Parameters loaded from %s",
            params_path,
        )

        return params

    except FileNotFoundError:
        logger.error(
            "Parameters file not found: %s",
            params_path,
        )
        raise

    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise

    except Exception as e:
        logger.error(
            "Unexpected error: %s",
            e,
        )
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df["content"] = df["content"].fillna("")

        logger.debug(
            "Data loaded from %s. Shape: %s",
            file_path,
            df.shape,
        )

        return df

    except pd.errors.ParserError as e:
        logger.error(
            "Failed to parse CSV: %s",
            e,
        )
        raise

    except Exception as e:
        logger.error(
            "Error loading data: %s",
            e,
        )
        raise


def apply_bow(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    try:
        vectorizer = CountVectorizer(
            max_features=max_features
        )

        X_train = train_data["content"]
        y_train = train_data["sentiment"]

        X_test = test_data["content"]
        y_test = test_data["sentiment"]

        X_train_bow = vectorizer.fit_transform(
            X_train
        )

        X_test_bow = vectorizer.transform(
            X_test
        )

        feature_names = vectorizer.get_feature_names_out()

        train_df = pd.DataFrame(
            X_train_bow.toarray(),
            columns=feature_names,
        )

        train_df["label"] = y_train.values

        test_df = pd.DataFrame(
            X_test_bow.toarray(),
            columns=feature_names,
        )

        test_df["label"] = y_test.values

        os.makedirs(
            "./models",
            exist_ok=True,
        )

        joblib.dump(
            vectorizer,
            "./models/vectorizer.pkl",
        )

        logger.debug(
            "BoW applied successfully. Vocabulary size: %d",
            len(feature_names),
        )

        logger.debug(
            "Train features shape: %s",
            train_df.shape,
        )

        logger.debug(
            "Test features shape: %s",
            test_df.shape,
        )

        return train_df, test_df

    except Exception as e:
        logger.error(
            "Error during BoW transformation: %s",
            e,
        )
        raise


def save_data(
    df: pd.DataFrame,
    file_path: str,
) -> None:

    try:
        directory = os.path.dirname(file_path)

        os.makedirs(
            directory,
            exist_ok=True,
        )

        df.to_csv(
            file_path,
            index=False,
        )

        logger.debug(
            "Data saved to %s",
            file_path,
        )

    except Exception as e:
        logger.error(
            "Error saving data: %s",
            e,
        )
        raise


def main():
    try:
        params = load_params(
            "params.yaml"
        )

        max_features = params[
            "feature_engineering"
        ]["max_features"]

        train_data = load_data(
            "./data/processed/train_processed.csv"
        )

        test_data = load_data(
            "./data/processed/test_processed.csv"
        )

        train_df, test_df = apply_bow(
            train_data,
            test_data,
            max_features,
        )

        save_data(
            train_df,
            "./data/features/train_bow.csv",
        )

        save_data(
            test_df,
            "./data/features/test_bow.csv",
        )

        logger.info(
            "Feature engineering completed successfully"
        )

    except Exception as e:
        logger.error(
            "Feature engineering failed: %s",
            e,
        )
        raise


if __name__ == "__main__":
    main()