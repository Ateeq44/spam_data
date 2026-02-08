"""
Spam Detection with Naive Bayes
===============================

This script implements a simple spam detector for text messages using
the Naive Bayes algorithm.  It uses the `spam_data.csv` file, which
contains labelled examples of spam and ham (non‑spam) messages.  The
steps performed are:

1. Load the dataset of messages and labels.
2. Split the data into training and testing sets.
3. Convert the text to numerical features using a bag‑of‑words
   representation (`CountVectorizer`).
4. Train a `MultinomialNB` classifier on the training data.
5. Evaluate the classifier on the test data and print accuracy.
6. Provide a command‑line interface for classifying new messages.

Usage:

```bash
python spam_classifier.py
```

Enter any text message when prompted, and the script will predict
whether it is spam or ham.

Dependencies:

* pandas
* scikit‑learn

Install required packages if necessary:

```bash
pip install pandas scikit-learn
```

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def load_data(path: str) -> tuple:
    """Load labelled text messages from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file containing columns `label` and `text`.

    Returns
    -------
    texts : list[str]
        List of message texts.
    labels : list[str]
        Corresponding list of labels ('spam' or 'ham').
    """
    df = pd.read_csv(path)
    return df['text'].tolist(), df['label'].tolist()


def train_and_evaluate(texts: list, labels: list) -> tuple:
    """Train a spam classifier and return the trained model and vectorizer.

    Splits the data into training and test sets, trains a
    MultinomialNB classifier and prints the accuracy on the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)
    # Evaluate
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained. Test set accuracy: {accuracy:.2f}")
    return classifier, vectorizer


def classify_message(classifier: MultinomialNB, vectorizer: CountVectorizer, message: str) -> str:
    """Return the predicted label ('spam' or 'ham') for a new message."""
    vec = vectorizer.transform([message])
    prediction = classifier.predict(vec)[0]
    return prediction


def main() -> None:
    # Load data
    try:
        texts, labels = load_data('spam_data.csv')
    except FileNotFoundError:
        print("Error: 'spam_data.csv' not found. Ensure the dataset is in the same directory.")
        return
    # Train and evaluate model
    classifier, vectorizer = train_and_evaluate(texts, labels)
    # CLI for user input
    print("\nEnter a message to classify. Type 'quit' to exit.")
    while True:
        msg = input("Message: ").strip()
        if msg.lower() in ['quit', 'exit']:
            print("Exiting...")
            break
        label = classify_message(classifier, vectorizer, msg)
        print(f"Prediction: {label}\n")


if __name__ == '__main__':
    main()