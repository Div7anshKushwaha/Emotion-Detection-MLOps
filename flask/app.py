import re
import string

import dagshub
import joblib
import mlflow
import pandas as pd

from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


app = Flask(__name__)


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


MODEL_URI = "models:/EmotionDetectionModel/1"

model = mlflow.pyfunc.load_model(MODEL_URI)
vectorizer = joblib.load("models/vectorizer.pkl")

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


def predict_emotion(text: str) -> str:
    processed_text = preprocess_text(text)

    features = vectorizer.transform(
        [processed_text]
    )

    input_data = pd.DataFrame(
        features.toarray(),
        columns=vectorizer.get_feature_names_out(),
    )

    prediction = model.predict(input_data)[0]

    return (
        "Happiness"
        if prediction == 1
        else "Sadness"
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": "EmotionDetectionModel",
        "version": 1,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({
                "error": "Text is required"
            }), 400

        emotion = predict_emotion(data["text"])

        return jsonify({
            "text": data["text"],
            "emotion": emotion,
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/predict-ui", methods=["POST"])
def predict_ui():
    text = request.form.get("text", "").strip()

    if not text:
        return render_template(
            "index.html",
            error="Please enter some text.",
        )

    try:
        emotion = predict_emotion(text)

        return render_template(
            "index.html",
            text=text,
            emotion=emotion,
        )

    except Exception as e:
        return render_template(
            "index.html",
            text=text,
            error=str(e),
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )