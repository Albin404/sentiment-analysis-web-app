# Viva Questions and Answers

1. **What is sentiment analysis?**  
   It is the process of identifying the opinion or emotion in text, such as positive, negative, or neutral.

2. **Why is preprocessing needed?**  
   Raw text contains inconsistent uppercase letters, punctuation, and common words that add little meaning. Cleaning creates more useful features.

3. **What is tokenization?**  
   Tokenization is splitting a sentence into individual words or tokens.

4. **What are stop words?**  
   Common words such as “the”, “is”, and “and” that usually provide little sentiment information.

5. **What is TF-IDF?**  
   Term Frequency–Inverse Document Frequency assigns a numerical importance score to each word. Words frequent in one document but uncommon in all documents get higher weight.

6. **Why use Multinomial Naive Bayes?**  
   It is simple, fast, and works well with word-count or TF-IDF features in text classification.

7. **What does “Naive” mean in Naive Bayes?**  
   It assumes features (words) are conditionally independent given the class. This assumption is simplified but often useful.

8. **What is the train-test split used here?**  
   80% of data trains the model and 20% unseen data tests it. This estimates how well the model generalizes.

9. **What is accuracy?**  
   The percentage of predictions that are correct: correct predictions divided by total predictions.

10. **What are precision and recall?**  
    Precision measures how many predicted examples of a class were correct. Recall measures how many actual examples of that class were found.

11. **What is a confusion matrix?**  
    A table showing actual labels against predicted labels. Its diagonal cells are correct predictions.

12. **Why save both the model and vectorizer?**  
    The vectorizer converts new text into the exact numerical feature format expected by the saved model.

13. **What happens if the dataset is imbalanced?**  
    The model may favor the majority class, making accuracy misleading. Balanced classes or class-aware evaluation help.

14. **Can Logistic Regression be used instead?**  
    Yes. It often performs strongly for TF-IDF text classification, but Naive Bayes was selected here for simplicity.

15. **What are limitations of this project?**  
    The small dataset and simple word features may not understand context, sarcasm, spelling errors, or complex sentences.
