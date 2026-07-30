"""Text cleaning helpers used by the training and prediction scripts."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negation changes sentiment: “not good” must not become just “good”.
STOP_WORDS = ENGLISH_STOP_WORDS.difference({"no", "nor", "not", "never"})


def preprocess_text(text: str) -> str:
    """Return lowercase, punctuation-free text without common stop words.

    Tokenization is done with a regular expression so this project does not
    require downloading NLTK data, which makes it easy to run in VS Code.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)       # remove punctuation/numbers
    tokens = re.findall(r"\b[a-z]+\b", text)    # tokenize
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)
