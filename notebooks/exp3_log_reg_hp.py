import re
import string

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split


dagshub.init(
    repo_owner="Div7anshKushwaha",
    repo_name="Emotion-Detection-MLOps",
    mlflow=True,
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Div7anshKushwaha/Emotion-Detection-MLOps.mlflow"
)


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
)


df = pd.read_csv(DATA_URL).drop(columns=["tweet_id"])

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(
        rf"[{re.escape(string.punctuation)}]",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


df = df[df["sentiment"].isin(["happiness", "sadness"])].copy()

df["sentiment"] = df["sentiment"].map(
    {
        "sadness": 0,
        "happiness": 1,
    }
)

df["content"] = df["content"].apply(preprocess_text)


X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["content"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"],
)


vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


mlflow.set_experiment("Logistic Regression Hyperparameter Tuning")


param_grid = {
    "C": [0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
}


with mlflow.start_run(
    run_name="Logistic Regression - Hyperparameter Tuning"
):

    mlflow.set_tags({
        "model": "Logistic Regression",
        "feature_engineering": "BoW",
        "dataset": "Tweet Emotions",
        "task": "Binary Emotion Classification",
    })

    mlflow.log_params({
        "test_size": 0.2,
        "random_state": 42,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "cv": 5,
        "scoring": "f1",
    })

    grid_search = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    for trial, (
        params,
        mean_score,
        std_score,
    ) in enumerate(
        zip(
            grid_search.cv_results_["params"],
            grid_search.cv_results_["mean_test_score"],
            grid_search.cv_results_["std_test_score"],
        ),
        start=1,
    ):

        with mlflow.start_run(
            run_name=f"Trial {trial}",
            nested=True,
        ):

            model = LogisticRegression(**params)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(
                    y_test,
                    y_pred,
                ),
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
            }

            mlflow.log_params(params)

            mlflow.log_metrics({
                "mean_cv_f1": mean_score,
                "std_cv_f1": std_score,
                **metrics,
            })

            mlflow.set_tag(
                "best_trial",
                str(params == grid_search.best_params_),
            )

            mlflow.sklearn.log_model(
                model,
                name="model",
            )

            print(
                f"Trial {trial} | "
                f"params={params} | "
                f"CV F1={mean_score:.4f} | "
                f"Test F1={metrics['f1_score']:.4f}"
            )

    best_params = grid_search.best_params_
    best_cv_f1 = grid_search.best_score_

    mlflow.log_params({
        "best_C": best_params["C"],
        "best_penalty": best_params["penalty"],
        "best_solver": best_params["solver"],
    })

    mlflow.log_metric(
        "best_cv_f1",
        best_cv_f1,
    )

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(
        grid_search.best_estimator_,
        name="best_model",
    )

    print()
    print("Best Parameters:", best_params)
    print(f"Best CV F1: {best_cv_f1:.4f}")


print("Hyperparameter tuning completed successfully.")