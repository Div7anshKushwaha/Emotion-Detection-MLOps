import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


MODEL_PATH = "models/model.pkl"
TEST_DATA_PATH = "data/features/test_bow.csv"

ACCURACY_THRESHOLD = 0.70
F1_THRESHOLD = 0.70


def load_model():
    return joblib.load(MODEL_PATH)


def load_test_data():
    data = pd.read_csv(TEST_DATA_PATH)
    X_test = data.drop(columns=["label"])
    y_test = data["label"]
    return X_test, y_test


def test_model_loads():
    model = load_model()

    assert model is not None


def test_model_prediction():
    model = load_model()
    X_test, _ = load_test_data()

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})


def test_model_accuracy():
    model = load_model()
    X_test, y_test = load_test_data()

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= ACCURACY_THRESHOLD


def test_model_f1_score():
    model = load_model()
    X_test, y_test = load_test_data()

    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)

    assert f1 >= F1_THRESHOLD