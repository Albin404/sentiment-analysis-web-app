# Sentiment Analysis Mini Project

A beginner-friendly B.Tech CSE (Data Science) mini project that classifies English reviews as **positive**, **negative**, or **neutral** using TF-IDF and Multinomial Naive Bayes.

## 1. Software needed

- Python 3.10 or newer (install from https://www.python.org/downloads/ and tick **Add Python to PATH**)
- Visual Studio Code
- VS Code extension: **Python** by Microsoft

## 2. Open and set up in VS Code (Windows)

1. Open the `SentimentAnalysisProject` folder in VS Code.
2. Open **Terminal > New Terminal**.
3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the same terminal and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

4. Install packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. In VS Code, select the `.venv` interpreter when prompted (or use `Ctrl+Shift+P` → **Python: Select Interpreter**).

## 3. Run the project

Train the model first:

```powershell
python train.py
```

It saves these files inside `models/`:

- `sentiment_model.joblib` — trained Naive Bayes model
- `tfidf_vectorizer.joblib` — fitted TF-IDF converter
- `confusion_matrix.png` — evaluation chart

Predict one review directly:

```powershell
python predict.py
```

Or start the menu application:

```powershell
python main.py
```

## 4. Run as a website (optional)

After installing the requirements, first train the model once with `python train.py`. Then run:

```powershell
streamlit run app.py
```

Your browser will open a local webpage. Type a review, click **Predict sentiment**, and the result is displayed on the page. You can also use **Train / retrain model** in the sidebar.

## 5. Dataset

`dataset/sentiment_reviews.csv` is a small, original classroom demonstration dataset with two columns:

| Column | Meaning |
|---|---|
| `review` | Text written by a customer/user |
| `sentiment` | Correct label: `positive`, `negative`, or `neutral` |

For a larger experiment, replace it with your own CSV using exactly the same headers and balanced labels. Keep at least 10 examples of each class to allow a stratified train/test split.

## 6. How it works

1. **Load CSV:** Pandas reads the review and sentiment columns.
2. **Preprocessing:** text becomes lowercase; punctuation/numbers are removed; words are tokenized; common English stop words such as `the`, `is`, and `and` are removed. Negation words such as `not` and `never` are kept because they change sentiment.
3. **TF-IDF:** important words become numbers. A word receives more weight when it is frequent in one review but not common across all reviews.
4. **Training:** Multinomial Naive Bayes learns which words commonly occur with each sentiment.
5. **Evaluation:** the held-out 20% test data gives accuracy, precision, recall, F1-score, and a confusion matrix.
6. **Prediction:** after evaluation, a final model is fitted on all labelled data, saved with joblib, and used to classify cleaned new reviews.

## 7. Expected sample output

Exact scores can vary after changing data or package versions. With the included classroom dataset, output will resemble:

```text
Model Accuracy: 100.00%

Classification Report:
              precision    recall  f1-score   support
    negative       1.00      1.00      1.00         2
     neutral       1.00      1.00      1.00         2
    positive       1.00      1.00      1.00         2

Saved model, vectorizer, and confusion-matrix image in models/.
```

Example prediction:

```text
Enter a review: The product quality is excellent and I love it
Predicted sentiment: Positive
```

## 8. Explanation of files

- `preprocess.py`: reusable cleaning function.
- `train.py`: loads data, creates TF-IDF features, trains/evaluates/saves the model.
- `predict.py`: loads saved files and predicts one review.
- `main.py`: easy menu that combines training and prediction.
- `app.py`: optional Streamlit website interface.
- `dataset/sentiment_reviews.csv`: labeled input data.
- `models/`: generated model and chart files after training.

## 9. Common issues

- **`python` is not recognized:** reinstall Python with “Add Python to PATH”, then reopen VS Code.
- **`ModuleNotFoundError`:** activate `.venv`, then run `python -m pip install -r requirements.txt`.
- **Model not found:** run `python train.py` once before prediction.
- **CSV error:** ensure the headers are exactly `review,sentiment` and all three labels have enough examples.
