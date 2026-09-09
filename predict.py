"""Use the saved model to predict sentiment for a new review."""
import json
from pathlib import Path
import joblib

from preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


def load_artifacts():
    """Load saved TF-IDF vectorizer and Naive Bayes model."""
    model_path = MODEL_DIR / "sentiment_model.joblib"
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError("Model not found. Run 'python train.py' first.")
    return joblib.load(model_path), joblib.load(vectorizer_path)


def predict_sentiment(review: str) -> str:
    """Return positive, negative, or neutral for one review."""
    model, vectorizer = load_artifacts()
    cleaned_review = preprocess_text(review)
    features = vectorizer.transform([cleaned_review])
    return model.predict(features)[0]


def predict_with_confidence(review: str) -> tuple[str, float]:
    """Return the predicted label and its probability percentage."""
    model, vectorizer = load_artifacts()
    cleaned_review = preprocess_text(review)
    features = vectorizer.transform([cleaned_review])
    probabilities = model.predict_proba(features)[0]
    best_index = probabilities.argmax()
    return model.classes_[best_index], float(probabilities[best_index] * 100)


def get_model_info() -> dict:
    """Return saved evaluation details for display in the web app."""
    info_path = MODEL_DIR / "model_info.json"
    if not info_path.exists():
        return {"selected_model": "Saved ML model", "accuracy": None, "weighted_f1": None}
    return json.loads(info_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    review = input("Enter a review: ").strip()
    if review:
        print("Predicted sentiment:", predict_sentiment(review).title())
    else:
        print("Please enter a non-empty review.")
