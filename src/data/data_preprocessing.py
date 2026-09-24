import logging
import os
import re
import string

import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    "transformation_errors.log"
)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text,
    )

    text = re.sub(
        r"\d+",
        "",
        text,
    )

    text = re.sub(
        rf"[{re.escape(string.punctuation)}]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()

        df["content"] = df["content"].fillna("")

        df["content"] = df["content"].apply(
            preprocess_text
        )

        df = df[df["content"].str.strip() != ""].copy()

        logger.debug(
            "Text normalization completed. Shape: %s",
            df.shape,
        )

        return df

    except Exception as e:
        logger.error(
            "Error during text normalization: %s",
            e,
        )
        raise


def main():
    try:
        train_data = pd.read_csv(
            "./data/raw/train.csv"
        )

        test_data = pd.read_csv(
            "./data/raw/test.csv"
        )

        logger.debug(
            "Train data loaded: %s",
            train_data.shape,
        )

        logger.debug(
            "Test data loaded: %s",
            test_data.shape,
        )

        train_processed_data = normalize_text(
            train_data
        )

        test_processed_data = normalize_text(
            test_data
        )

        data_path = os.path.join(
            "./data",
            "processed",
        )

        os.makedirs(
            data_path,
            exist_ok=True,
        )

        train_processed_data.to_csv(
            os.path.join(
                data_path,
                "train_processed.csv",
            ),
            index=False,
        )

        test_processed_data.to_csv(
            os.path.join(
                data_path,
                "test_processed.csv",
            ),
            index=False,
        )

        logger.info(
            "Processed data saved to %s",
            data_path,
        )

    except Exception as e:
        logger.error(
            "Data preprocessing failed: %s",
            e,
        )
        raise


if __name__ == "__main__":
    main()