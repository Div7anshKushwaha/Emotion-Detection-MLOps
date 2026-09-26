import json
import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    "model_evaluation_errors.log"
)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path: str):
    try:
        model = joblib.load(file_path)
        logger.debug("Model loaded from %s", file_path)
        return model

    except FileNotFoundError:
        logger.error("Model file not found: %s", file_path)
        raise

    except Exception as e:
        logger.error("Error loading model: %s", e)
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
        logger.error("Failed to parse CSV: %s", e)
        raise

    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


def evaluate_model(model, X_test, y_test) -> dict:
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "auc": roc_auc_score(
                y_test,
                y_pred_proba,
            ),
        }

        logger.debug("Model evaluation completed")

        return metrics

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True,
    )

    with open(file_path, "w") as file:
        json.dump(metrics, file, indent=4)

    logger.debug("Metrics saved to %s", file_path)


def save_model_info(
    run_id: str,
    model_id: str,
    file_path: str,
) -> None:

    model_info = {
        "run_id": run_id,
        "model_id": model_id,
    }

    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True,
    )

    with open(file_path, "w") as file:
        json.dump(model_info, file, indent=4)

    logger.debug(
        "Model information saved to %s",
        file_path,
    )


def main():
    try:
        mlflow.set_experiment(
            "Emotion Detection - DVC Pipeline"
        )

        with mlflow.start_run(
            run_name="Logistic Regression - Evaluation"
        ) as run:

            model = load_model(
                "./models/model.pkl"
            )

            test_data = load_data(
                "./data/features/test_bow.csv"
            )

            X_test = test_data.drop(
                columns=["label"]
            )

            y_test = test_data["label"]

            metrics = evaluate_model(
                model,
                X_test,
                y_test,
            )

            save_metrics(
                metrics,
                "reports/metrics.json",
            )

            mlflow.log_metrics(metrics)

            mlflow.log_params(
                model.get_params()
            )

            logged_model = mlflow.sklearn.log_model(
                model,
                name="model",
            )

            save_model_info(
                run.info.run_id,
                logged_model.model_id,
                "reports/model_info.json",
            )

            mlflow.log_artifact(
                "reports/metrics.json"
            )

            mlflow.log_artifact(
                "reports/model_info.json"
            )

            mlflow.log_artifact(
                __file__,
                artifact_path="source",
            )

            if os.path.exists(
                "model_evaluation_errors.log"
            ):
                mlflow.log_artifact(
                    "model_evaluation_errors.log"
                )

            logger.info(
                "Model evaluation completed successfully"
            )
            logger.info(
                "Metrics: %s",
                metrics,
            )

    except Exception as e:
        logger.error(
            "Model evaluation failed: %s",
            e,
        )
        raise


if __name__ == "__main__":
    main()