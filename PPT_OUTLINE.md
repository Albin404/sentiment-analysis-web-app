# PPT Presentation Outline (9 Slides)

## Slide 1 — Title

Sentiment Analysis of Customer Reviews Using TF-IDF and Naive Bayes. Include name, roll number, department, year, guide, and college.

## Slide 2 — Problem Statement

Online reviews are numerous and difficult to read manually. The project automatically classifies a review as positive, negative, or neutral.

## Slide 3 — Objectives

Mention three-class classification, preprocessing, TF-IDF, model evaluation, and new-review prediction.

## Slide 4 — Technologies Used

Python, VS Code, Pandas, scikit-learn, Joblib, Matplotlib, Seaborn, CSV dataset.

## Slide 5 — System Workflow

CSV Dataset → Text Preprocessing → TF-IDF → Naive Bayes Model → Evaluation / New Prediction → Saved Model.

## Slide 6 — Text Preprocessing

Show a small example: “This Product is AMAZING!!!” → lowercase → punctuation removal → tokens → stop-word removal → “product amazing”.

## Slide 7 — Model and Evaluation

Explain that Multinomial Naive Bayes calculates probabilities for text classes. Show accuracy, classification report, and a screenshot of `models/confusion_matrix.png` after running the program.

## Slide 8 — Demonstration

Show the terminal commands: `python train.py`, then `python predict.py`; include one positive and one negative review prediction.

## Slide 9 — Conclusion and Future Scope

Summarize the working pipeline. Mention larger datasets, Logistic Regression/SVM, web interface, and multilingual analysis as future work.
