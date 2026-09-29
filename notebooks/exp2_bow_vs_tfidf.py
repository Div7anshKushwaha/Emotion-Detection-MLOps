import re
import string

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


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
    {"sadness": 0, "happiness": 1}
)
df["content"] = df["content"].apply(preprocess_text)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["content"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"],
)


vectorizers = {
    "BoW": CountVectorizer(),
    "TF-IDF": TfidfVectorizer(),
}


algorithms = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(
        random_state=42,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42,
    ),
}


mlflow.set_experiment("BoW vs TF-IDF v2")


with mlflow.start_run(run_name="All Experiments"):

    mlflow.set_tags({
        "dataset": "Tweet Emotions",
        "task": "Binary Emotion Classification",
    })

    mlflow.log_params({
        "test_size": 0.2,
        "random_state": 42,
        "train_samples": len(X_train_text),
        "test_samples": len(X_test_text),
    })

    for vec_name, vectorizer in vectorizers.items():

        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        for algo_name, model in algorithms.items():

            with mlflow.start_run(
                run_name=f"{algo_name} with {vec_name}",
                nested=True,
            ):

                mlflow.set_tags({
                    "algorithm": algo_name,
                    "feature_engineering": vec_name,
                })

                mlflow.log_params({
                    "algorithm": algo_name,
                    "vectorizer": vec_name,
                    "vocabulary_size": len(
                        vectorizer.vocabulary_
                    ),
                })

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

                mlflow.log_metrics(metrics)

                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    skops_trusted_types=[
                        "sklearn.tree._tree.Tree"
                    ],
                )

                mlflow.log_artifact(
                    __file__,
                    artifact_path="source",
                )

                print(
                    f"{algo_name} + {vec_name}: "
                    f"accuracy={metrics['accuracy']:.4f}, "
                    f"f1={metrics['f1_score']:.4f}"
                )


print("All experiments completed successfully.")