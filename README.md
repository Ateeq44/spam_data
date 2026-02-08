<h1>Spam Detection Project</h1>
  <p>
    This project provides a simple spam detection system for SMS or text messages using the <strong>Multinomial Naive Bayes</strong> algorithm. It
    includes a command‑line classifier and an interactive Streamlit dashboard for classifying messages as <em>spam</em> or <em>ham</em> (not spam).
    The dataset <code>spam_data.csv</code> is a small synthetic collection of labelled messages.
  </p>

  <h2>Project Structure</h2>
  <table>
    <thead>
      <tr><th>File</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr>
        <td><code>spam_classifier.py</code></td>
        <td>Command‑line script that trains a Naive Bayes model, prints accuracy and classifies user messages.</td>
      </tr>
      <tr>
        <td><code>spam_dashboard.py</code></td>
        <td>Streamlit dashboard for classifying messages interactively, summarising the dataset and showing examples.</td>
      </tr>
      <tr>
        <td><code>spam_data.csv</code></td>
        <td>Synthetic dataset containing two columns: <code>label</code> (spam or ham) and <code>text</code> (message content).</td>
      </tr>
    </tbody>
  </table>

  <h2>Features</h2>
  <h3>CLI Script</h3>
  <p>The command‑line script performs the following steps:</p>
  <ol>
    <li>Reads <code>spam_data.csv</code> into memory and splits it into training and testing sets (80/20 split).</li>
    <li>Converts messages into numerical features using a <strong>bag‑of‑words</strong> representation via <code>CountVectorizer</code>.</li>
    <li>Trains a <strong>Multinomial Naive Bayes</strong> classifier on the training data and prints the test accuracy.</li>
    <li>Provides a prompt where you can enter a new message and see whether it is predicted as spam or ham.</li>
  </ol>
  <p>Run the script with:</p>
  <pre><code>python spam_classifier.py</code></pre>
  <p>The script will train the model, display accuracy and allow you to classify messages. Type <code>quit</code> to exit.</p>

  <h3>Streamlit Dashboard</h3>
  <p>The dashboard offers an easy‑to‑use interface:</p>
  <ul>
    <li><strong>Spam classifier</strong> – Enter any message in a text area; the model instantly predicts whether it is spam or ham.</li>
    <li><strong>Dataset summary</strong> – Displays counts of spam and ham messages in tabular form for a quick overview.</li>
    <li><strong>Sample messages</strong> – Shows a few example spam and ham messages from the dataset to illustrate typical patterns.</li>
  </ul>
  <p>To launch the dashboard, run:</p>
  <pre><code>python -m streamlit run spam_dashboard.py</code></pre>

  <h2>Installation</h2>
  <ol>
    <li>Ensure you have Python 3.8 or higher installed.</li>
    <li>Install dependencies using pip:
      <pre><code>python -m pip install streamlit pandas scikit-learn matplotlib</code></pre>
    </li>
  </ol>

  <h2>Customising the Dataset</h2>
  <p>You can supply your own labelled messages by replacing <code>spam_data.csv</code> with a CSV file containing two columns:</p>
  <ul>
    <li><code>label</code> – Must contain the string <code>spam</code> or <code>ham</code> for each message.</li>
    <li><code>text</code> – The content of the message.</li>
  </ul>
  <p>Keep the column names exactly as above or modify the code accordingly. The model expects a binary classification (spam vs ham) and may not perform well on languages other than English unless retrained on appropriate data.</p>

  <h2>Model Considerations</h2>
  <p>The <strong>Multinomial Naive Bayes</strong> classifier is a baseline algorithm for text classification tasks. It works well for simple datasets but may underperform on more complex or unbalanced data. For improved results, consider using TF‑IDF features, logistic regression, support vector machines or deep learning models such as recurrent neural networks.</p>
