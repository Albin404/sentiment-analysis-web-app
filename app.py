"""Simple web interface for the sentiment analysis project.

Run with: streamlit run app.py
"""
import streamlit as st

from predict import predict_sentiment
from train import train_model


st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")
st.title("💬 Sentiment Analysis of Reviews")
st.write("Enter an English review to classify it as Positive, Negative, or Neutral.")

with st.sidebar:
    st.header("Model setup")
    if st.button("Train / retrain model"):
        with st.spinner("Training the model..."):
            train_model()
        st.success("Model trained and saved successfully.")

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
            sentiment = predict_sentiment(review)
            display = sentiment.title()
            if sentiment == "positive":
                st.success(f"Predicted sentiment: {display} 😊")
            elif sentiment == "negative":
                st.error(f"Predicted sentiment: {display} 😟")
            else:
                st.info(f"Predicted sentiment: {display} 😐")
        except FileNotFoundError:
            st.warning("Please click ‘Train / retrain model’ in the sidebar first.")

st.divider()
st.caption("Model: TF-IDF + Multinomial Naive Bayes | Labels: Positive, Negative, Neutral")
