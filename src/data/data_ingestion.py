import logging
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.log")
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

        logger.debug("Parameters loaded from %s", params_path)
        return params

    except FileNotFoundError:
        logger.error("Parameters file not found: %s", params_path)
        raise

    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)

        logger.debug("Data loaded from %s", data_url)
        logger.debug("Dataset shape: %s", df.shape)

        return df

    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s", e)
        raise

    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns=["tweet_id"])

        df = df[
            df["sentiment"].isin(["happiness", "sadness"])
        ].copy()

        df["sentiment"] = df["sentiment"].map({
            "happiness": 1,
            "sadness": 0,
        })

        logger.debug(
            "Preprocessed dataset shape: %s",
            df.shape,
        )

        return df

    except KeyError as e:
        logger.error("Missing column: %s", e)
        raise

    except Exception as e:
        logger.error(
            "Error during preprocessing: %s",
            e,
        )
        raise


def save_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    data_path: str,
) -> None:
    try:
        raw_data_path = os.path.join(
            data_path,
            "raw",
        )

        os.makedirs(
            raw_data_path,
            exist_ok=True,
        )

        train_data.to_csv(
            os.path.join(
                raw_data_path,
                "train.csv",
            ),
            index=False,
        )

        test_data.to_csv(
            os.path.join(
                raw_data_path,
                "test.csv",
            ),
            index=False,
        )

        logger.debug(
            "Train and test data saved to %s",
            raw_data_path,
        )

    except Exception as e:
        logger.error(
            "Error saving data: %s",
            e,
        )
        raise


def main():
    try:
        params = load_params("params.yaml")

        test_size = params["data_ingestion"]["test_size"]
        random_state = params["data_ingestion"]["random_state"]

        data_url = (
            "https://raw.githubusercontent.com/"
            "campusx-official/jupyter-masterclass/"
            "main/tweet_emotions.csv"
        )

        df = load_data(data_url)

        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=random_state,
            stratify=final_df["sentiment"],
        )

        save_data(
            train_data,
            test_data,
            "./data",
        )

        logger.info("Data ingestion completed successfully")

    except Exception as e:
        logger.error(
            "Data ingestion failed: %s",
            e,
        )
        raise


if __name__ == "__main__":
    main()