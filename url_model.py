# url_model.py -- simple feature-based classifier (scikit-learn)
import pandas as pd
from urllib.parse import urlparse
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Example feature extractor
def extract_url_features(url):
    features = {}
    parsed = urlparse(url)
    features['len'] = len(url)
    features['num_dots'] = url.count('.')
    features['has_at'] = '@' in url
    features['has_ip'] = bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))
    features['path_len'] = len(parsed.path)
    features['query_len'] = len(parsed.query)
    return features

# load your dataset as a CSV with columns: url,label (1=phish,0=benign)
# Ensure you have a file named 'urls.csv' in the same directory.
df = pd.read_csv('urls.csv')
X = df['url'].apply(extract_url_features).tolist()
y = df['label'].values

vec = DictVectorizer(sparse=False)
X_vec = vec.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_vec, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print(classification_report(y_te, clf.predict(X_te)))
