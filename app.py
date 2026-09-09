"""Simple and reliable Streamlit interface for sentiment analysis."""
from pathlib import Path

import pandas as pd
import streamlit as st

from predict import get_model_info, predict_with_confidence
from train import train_model


st.set_page_config(page_title="FeelFinder | Sentiment Analysis", page_icon="💬")

DATASET_PATH = Path(__file__).parent / "dataset" / "sentiment_reviews.csv"


@st.cache_data
def load_reviews() -> pd.DataFrame:
    """Load the labelled reviews and create display-friendly star ratings."""
    reviews = pd.read_csv(DATASET_PATH)
    ratings = {
        "positive": [5, 5, 4],
        "neutral": [3],
        "negative": [1, 2, 1],
    }
    reviews["rating"] = [
        ratings[sentiment][index % len(ratings[sentiment])]
        for index, sentiment in enumerate(reviews["sentiment"])
    ]
    reviews["stars"] = reviews["rating"].apply(lambda rating: "★" * rating + "☆" * (5 - rating))
    return reviews

st.title("💬 FeelFinder")
st.write("Enter a review to classify it as **Positive**, **Negative**, or **Neutral**.")

with st.sidebar:
    st.header("Model details")
    model_info = get_model_info()
    st.write("Selected model:")
    st.success(model_info["selected_model"])
    if model_info["weighted_f1"] is not None:
        st.caption(f"Weighted F1-score: {model_info['weighted_f1']}%")

    show_reviews = st.toggle(
        "Show review explorer",
        value=True,
        help="Show or hide the table of dataset reviews.",
    )

    if st.button("Train / retrain model"):
        with st.spinner("Comparing Naive Bayes and Logistic Regression..."):
            result = train_model()
        st.success(f"Selected: {result['selected_model']}")

review = st.text_area(
    "Enter your review",
    placeholder="Example: The product quality is excellent and I love it.",
    height=130,
)

if st.button("Predict sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            sentiment, confidence = predict_with_confidence(review)
            if sentiment == "positive":
                st.success(f"Predicted sentiment: Positive 😊")
            elif sentiment == "negative":
                st.error(f"Predicted sentiment: Negative 😟")
            else:
                st.info(f"Predicted sentiment: Neutral 😐")
            st.metric("Model confidence", f"{confidence:.1f}%")
        except FileNotFoundError:
            st.warning("Please click ‘Train / retrain model’ in the sidebar first.")

st.caption("TF-IDF • Naive Bayes vs Logistic Regression • Best model selected automatically")

if show_reviews:
    st.divider()
    st.subheader("Customer review explorer")
    st.write("Choose a category to view the labelled reviews and their star ratings.")

    selected_category = st.segmented_control(
        "Review category",
        ["All reviews", "Positive", "Negative", "Neutral"],
        default="All reviews",
        required=True,
        key="review_category",
    )

    reviews = load_reviews()
    if selected_category != "All reviews":
        reviews = reviews[reviews["sentiment"] == selected_category.lower()]

    display_reviews = reviews.rename(
        columns={"review": "Review", "sentiment": "Sentiment", "rating": "Rating", "stars": "Stars"}
    )[["Review", "Stars", "Rating", "Sentiment"]]

    st.caption(f"Showing {len(display_reviews)} review(s)")
    st.dataframe(display_reviews, hide_index=True, width="stretch")
