import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- Step 1: Load and prepare URL data ---
print("Training URL Classifier...")
urls_df = pd.read_csv('urls.csv') # Assuming 'urls.csv' has 'url' and 'label' columns
X_url = urls_df['url']
y_url = urls_df['label']

# Convert URLs to numerical features (this is a simplified example)
# You might have a more complex feature extraction function in 'feature_extractor.py'
vectorizer_url = TfidfVectorizer(max_features=1000)
X_url_features = vectorizer_url.fit_transform(X_url)

# Train the URL classifier
url_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
url_classifier.fit(X_url_features, y_url)

# Save the trained URL classifier and its vectorizer
joblib.dump(url_classifier, 'url_classifier.joblib')
joblib.dump(vectorizer_url, 'url_vectorizer.joblib') # Save the vectorizer too!
print("URL classifier and vectorizer saved as url_classifier.joblib and url_vectorizer.joblib")


# --- Step 2: Load and prepare text data ---
print("\nTraining Text Classifier...")
emails_df = pd.read_csv('emails.csv') # Assuming 'emails.csv' has 'text' and 'label' columns
X_text = emails_df['text']
y_text = emails_df['label']

# Convert text to numerical features
vectorizer_text = TfidfVectorizer(max_features=2000)
X_text_features = vectorizer_text.fit_transform(X_text)

# Train the text classifier
text_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
text_classifier.fit(X_text_features, y_text)

# Save the trained text classifier and its vectorizer
joblib.dump(text_classifier, 'text_classifier.joblib')
joblib.dump(vectorizer_text, 'text_vectorizer.joblib')
print("Text classifier and vectorizer saved as text_classifier.joblib and text_vectorizer.joblib")
