import json
import logging

import dagshub
import mlflow
from mlflow.tracking import MlflowClient


dagshub.init(
    repo_owner="Div7anshKushwaha",
    repo_name="Emotion-Detection-MLOps",
    mlflow=True,
)

mlflow.set_tracking_uri(
    "https://dagshub.com/"
    "Div7anshKushwaha/"
    "Emotion-Detection-MLOps.mlflow"
)


logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    "model_registration_errors.log"
)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)

        logger.debug(
            "Model info loaded from %s",
            file_path,
        )

        return model_info

    except FileNotFoundError:
        logger.error(
            "Model info file not found: %s",
            file_path,
        )
        raise

    except Exception as e:
        logger.error(
            "Error loading model info: %s",
            e,
        )
        raise


def register_model(
    model_name: str,
    model_info: dict,
) -> None:

    try:
        model_uri = (
            f"models:/{model_info['model_id']}"
        )

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        client = MlflowClient()

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
        )

        logger.info(
            "Model registered successfully: "
            "%s version %s",
            model_name,
            model_version.version,
        )

        logger.info(
            "Model transitioned to Staging"
        )

    except Exception as e:
        logger.error(
            "Model registration failed: %s",
            e,
        )
        raise


def main():
    try:
        model_info = load_model_info(
            "reports/model_info.json"
        )

        register_model(
            model_name="EmotionDetectionModel",
            model_info=model_info,
        )

    except Exception as e:
        logger.error(
            "Model registration process failed: %s",
            e,
        )
        raise


if __name__ == "__main__":
    main()