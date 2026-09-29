import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

dagshub_token = os.getenv("DAGSHUB_PAT")
dagshub_username = os.getenv("DAGSHUB_USERNAME")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

if not dagshub_username:
    raise EnvironmentError("DAGSHUB_USERNAME environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(
    "https://dagshub.com/Div7anshKushwaha/Emotion-Detection-MLOps.mlflow"
)

logger = logging.getLogger("model_promotion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    "model_promotion_errors.log"
)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

MODEL_NAME = "EmotionDetectionModel"


def promote_latest_model():
    try:
        client = MlflowClient()

        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        if not versions:
            raise ValueError(
                f"No registered versions found for {MODEL_NAME}"
            )

        latest_version = max(
            versions,
            key=lambda version: int(version.version),
        )

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version.version,
            stage="Production",
        )

        logger.info(
            "Model version %s promoted to Production",
            latest_version.version,
        )

    except Exception as e:
        logger.error(
            "Model promotion failed: %s",
            e,
        )
        raise


if __name__ == "__main__":
    promote_latest_model()