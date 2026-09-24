import logging
import os

import joblib
import pandas as pd
import yaml

from sklearn.linear_model import LogisticRegression


logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    "model_building_errors.log"
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
        logger.error(
            "YAML error: %s",
            e,
        )
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


def train_model(
    X_train,
    y_train,
    params: dict,
) -> LogisticRegression:

    try:
        model = LogisticRegression(
            C=params["C"],
            solver=params["solver"],
            penalty=params["penalty"],
            max_iter=params["max_iter"],
        )

        model.fit(
            X_train,
            y_train,
        )

        logger.debug(
            "Model training completed"
        )

        return model

    except Exception as e:
        logger.error(
            "Error during model training: %s",
            e,
        )
        raise


def save_model(
    model,
    file_path: str,
) -> None:

    try:
        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True,
        )

        joblib.dump(
            model,
            file_path,
        )

        logger.debug(
            "Model saved to %s",
            file_path,
        )

    except Exception as e:
        logger.error(
            "Error saving model: %s",
            e,
        )
        raise


def main():
    try:
        params = load_params(
            "params.yaml"
        )

        model_params = params[
            "model_building"
        ]

        train_data = load_data(
            "./data/features/train_bow.csv"
        )

        X_train = train_data.drop(
            columns=["label"]
        )

        y_train = train_data["label"]

        model = train_model(
            X_train,
            y_train,
            model_params,
        )

        save_model(
            model,
            "models/model.pkl",
        )

        logger.info(
            "Model building completed successfully"
        )

    except Exception as e:
        logger.error(
            "Model building failed: %s",
            e,
        )
        raise


if __name__ == "__main__":
    main()