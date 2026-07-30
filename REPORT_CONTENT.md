# Mini Project Report Content

## Title

Sentiment Analysis of Customer Reviews Using TF-IDF and Naive Bayes

## Abstract

This project builds a sentiment analysis system to identify whether a user review is positive, negative, or neutral. The system reads reviews from a CSV dataset, cleans the text through lowercase conversion, punctuation removal, stop-word removal, and tokenization. TF-IDF converts the processed text into numerical feature vectors. A Multinomial Naive Bayes classifier is trained using these features. The system evaluates its performance with accuracy, a classification report, and a confusion matrix. It also accepts a new review from the user and predicts its sentiment. The trained model and vectorizer are saved with joblib for later use.

## Objectives

- Build a simple three-class sentiment classifier.
- Understand text preprocessing and TF-IDF feature extraction.
- Train and evaluate a machine learning model.
- Save and reuse a trained model for new predictions.

## Tools and Technologies

Python, VS Code, Pandas, scikit-learn, Matplotlib, Seaborn, and Joblib.

## Methodology

The CSV dataset is loaded with Pandas. Missing reviews are removed. Each review is converted to lowercase, punctuation and numbers are removed, the text is split into tokens, and English stop words are removed. TF-IDF creates a numeric matrix from the cleaned text. The data is divided into 80% training data and 20% testing data using stratified splitting, so every sentiment class is represented. Multinomial Naive Bayes is trained on the training features. The test predictions are compared to actual labels to calculate accuracy, precision, recall, F1-score, and a confusion matrix. Finally, joblib saves the model and vectorizer.

## Algorithm

1. Read `review` and `sentiment` from CSV.
2. Clean every review.
3. Split data into training and testing sets.
4. Fit TF-IDF on training reviews.
5. Transform training and testing reviews into numerical vectors.
6. Train Multinomial Naive Bayes.
7. Evaluate the test predictions.
8. Save model and vectorizer.
9. For new input, repeat cleaning and TF-IDF transformation, then predict the label.

## Results and Discussion

The program displays an accuracy percentage and detailed classification metrics. The confusion matrix visualizes correct and incorrect predictions for positive, negative, and neutral reviews. Positive and negative sentences with clear words such as “excellent” and “terrible” are typically easier to classify. Neutral reviews can be harder because they may contain fewer opinion words. Results depend on the size, balance, and quality of the dataset.

## Conclusion

The project successfully demonstrates an end-to-end natural language processing workflow: data loading, preprocessing, feature extraction, classification, evaluation, prediction, and model persistence. It is simple enough for laboratory learning and can be extended with a larger real-world dataset or more advanced models.

## Future Scope

- Use a larger dataset from product-review websites.
- Compare Logistic Regression, SVM, and deep learning models.
- Add a Streamlit web interface.
- Support multiple languages and emoji handling.

## References

- scikit-learn documentation: https://scikit-learn.org/
- Pandas documentation: https://pandas.pydata.org/
