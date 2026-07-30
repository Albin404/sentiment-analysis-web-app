"""Train and evaluate the sentiment classifier."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "sentiment_reviews.csv"
MODEL_DIR = BASE_DIR / "models"


def train_model():
    """Read data, create TF-IDF features, train Naive Bayes, and save files."""
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

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    labels = ["positive", "negative", "neutral"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print("Confusion Matrix (rows=actual, columns=predicted):\n", matrix)

    MODEL_DIR.mkdir(exist_ok=True)
    # After evaluation, train one final version on all available data. This is
    # the version saved for real predictions, so it can learn from every
    # labelled review instead of only the 80% training split.
    final_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    all_features = final_vectorizer.fit_transform(data["clean_review"])
    final_model = MultinomialNB()
    final_model.fit(all_features, data["sentiment"])
    joblib.dump(final_model, MODEL_DIR / "sentiment_model.joblib")
    joblib.dump(final_vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted sentiment")
    plt.ylabel("Actual sentiment")
    plt.title("Sentiment Analysis Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("\nSaved model, vectorizer, and confusion-matrix image in models/.")


if __name__ == "__main__":
    train_model()
