"""
Spam Detection Dashboard
=======================

This Streamlit application provides an interactive interface for
classifying text messages as spam or ham (non‑spam) using a
Multinomial Naive Bayes classifier trained on the provided
`spam_data.csv` dataset.  It also displays some simple analytics about
the dataset.

Features:

* **Spam classifier** – Enter any message in the text box, and the
  model will predict whether it is spam or ham.
* **Dataset summary** – Displays the number of spam and ham messages
  in tabular form.
* **Example messages** – Shows a few sample spam and ham messages
  from the dataset to help understand typical patterns.

Run the app with:

```bash
streamlit run spam_dashboard.py
```

Dependencies:

* streamlit
* pandas
* scikit‑learn
* matplotlib

"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(data: pd.DataFrame) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)
    acc = accuracy_score(y_test, classifier.predict(X_test_vec))
    return classifier, vectorizer, acc


def classify_text(msg: str, classifier: MultinomialNB, vectorizer: CountVectorizer) -> str:
    vec = vectorizer.transform([msg])
    return classifier.predict(vec)[0]


def main() -> None:
    st.set_page_config(page_title="Spam Detection Dashboard", layout="wide")
    st.title("Spam Detection Dashboard")
    try:
        data = load_data('spam_data.csv')
    except FileNotFoundError:
        st.error("Dataset 'spam_data.csv' not found.")
        return
    # Train the model
    classifier, vectorizer, acc = train_model(data)
    st.write(f"Model accuracy on test set: {acc:.2f}")
    # Input box for classification
    st.subheader("Classify a Message")
    user_msg = st.text_area("Enter message text:", height=100)
    if user_msg:
        prediction = classify_text(user_msg, classifier, vectorizer)
        if prediction == 'spam':
            st.error("This message is predicted to be SPAM.")
        else:
            st.success("This message is predicted to be HAM (not spam).")
    # Dataset summary
    st.subheader("Dataset Summary")
    counts = data['label'].value_counts()
    summary_df = counts.rename(index={'ham': 'Ham (non‑spam)', 'spam': 'Spam'}).reset_index()
    summary_df.columns = ['Label', 'Count']
    # Display counts without charts for a cleaner professional look
    st.table(summary_df)
    # Show some example messages
    st.subheader("Sample Messages")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Spam examples:**")
        for msg in data[data['label'] == 'spam']['text'].head(3):
            st.write(f"- {msg}")
    with col2:
        st.write("**Ham examples:**")
        for msg in data[data['label'] == 'ham']['text'].head(3):
            st.write(f"- {msg}")


if __name__ == '__main__':
    main()