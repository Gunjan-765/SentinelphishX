# email_model.py -- TF-IDF + Naive Bayes classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Step 1: Load the dataset
# Make sure your 'emails.csv' file is in the same directory
try:
    df = pd.read_csv('emails.csv')
except FileNotFoundError:
    print("Error: The file 'emails.csv' was not found.")
    print("Please create the file with 'text' and 'label' columns.")
    exit()

X = df['text']  # The email texts
y = df['label'] # The labels (0 or 1)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create the TF-IDF + Multinomial Naive Bayes pipeline
# The pipeline automates the steps of converting text to numbers and then training the model
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
    ('clf', MultinomialNB())
])

# Step 4: Train the model
print("Training the model...")
pipe.fit(X_train, y_train)
print("Model training complete.\n")

# Step 5: Evaluate the model
print("Evaluating the model on the test data...")
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

# Example: Make a prediction on a new email
new_email = ["Urgent: Your account has been suspended."]
prediction = pipe.predict(new_email)
if prediction[0] == 1:
    print("Prediction for new email: This is a phishing email.")
else:
    print("Prediction for new email: This is a safe email.")
