"""Train, compare, and save sentiment classifiers."""
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "sentiment_reviews.csv"
MODEL_DIR = BASE_DIR / "models"


def train_model():
    """Train two ML models and save the one with the best weighted F1-score."""
    data = pd.read_csv(DATA_FILE)
    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(data.columns):
        raise ValueError("CSV must contain columns named: review, sentiment")

    data = data.dropna(subset=["review", "sentiment"]).copy()
    data["clean_review"] = data["review"].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_review"], data["sentiment"], test_size=0.20,
        random_state=42, stratify=data["sentiment"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    labels = ["positive", "negative", "neutral"]
    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }
    results = []
    trained_models = {}
    for name, model in candidates.items():
        model.fit(X_train_tfidf, y_train)
        predictions = model.predict(X_test_tfidf)
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "weighted_f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        })
        trained_models[name] = (model, predictions)

    best_result = max(results, key=lambda item: (item["weighted_f1"], item["accuracy"]))
    best_name = best_result["model"]
    best_predictions = trained_models[best_name][1]
    matrix = confusion_matrix(y_test, best_predictions, labels=labels)

    print("\nModel comparison:")
    for result in results:
        print(
            f"- {result['model']}: Accuracy {result['accuracy'] * 100:.2f}% | "
            f"Weighted F1 {result['weighted_f1'] * 100:.2f}%"
        )
    print(f"\nSelected model: {best_name}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, best_predictions, labels=labels, zero_division=0))
    print("Confusion Matrix (rows=actual, columns=predicted):\n", matrix)

    MODEL_DIR.mkdir(exist_ok=True)
    # After comparison, train the selected model on all data. This is the
    # version used by the website for real predictions.
    final_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    all_features = final_vectorizer.fit_transform(data["clean_review"])
    final_model = candidates[best_name]
    final_model.fit(all_features, data["sentiment"])
    joblib.dump(final_model, MODEL_DIR / "sentiment_model.joblib")
    joblib.dump(final_vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    pd.DataFrame(results).to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    model_info = {
        "selected_model": best_name,
        "accuracy": round(best_result["accuracy"] * 100, 2),
        "weighted_f1": round(best_result["weighted_f1"] * 100, 2),
    }
    (MODEL_DIR / "model_info.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted sentiment")
    plt.ylabel("Actual sentiment")
    plt.title("Sentiment Analysis Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("\nSaved selected model, vectorizer, comparison, and confusion matrix in models/.")
    return model_info


if __name__ == "__main__":
    train_model()
