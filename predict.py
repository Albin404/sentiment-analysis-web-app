"""Use the saved model to predict sentiment for a new review."""
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


if __name__ == "__main__":
    review = input("Enter a review: ").strip()
    if review:
        print("Predicted sentiment:", predict_sentiment(review).title())
    else:
        print("Please enter a non-empty review.")
