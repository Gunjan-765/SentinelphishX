import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load the saved vectorizers ---
# These must be loaded to ensure consistency with how the training data was processed.
try:
    loaded_url_vectorizer = joblib.load('url_vectorizer.joblib')
    print("Feature Extractor: URL vectorizer loaded.")
except FileNotFoundError:
    print("Feature Extractor Error: 'url_vectorizer.joblib' not found. Please train your models first.")
    loaded_url_vectorizer = None # Set to None to handle gracefully
except Exception as e:
    print(f"Feature Extractor Error loading URL vectorizer: {e}")
    loaded_url_vectorizer = None

try:
    loaded_text_vectorizer = joblib.load('text_vectorizer.joblib')
    print("Feature Extractor: Text vectorizer loaded.")
except FileNotFoundError:
    print("Feature Extractor Error: 'text_vectorizer.joblib' not found. Please train your models first.")
    loaded_text_vectorizer = None
except Exception as e:
    print(f"Feature Extractor Error loading Text vectorizer: {e}")
    loaded_text_vectorizer = None


def extract_url_features(url):
    """
    Extracts numerical features from a given URL using the pre-trained URL vectorizer.

    Args:
        url (str): The URL string to extract features from.
        # Note: The vectorizer is loaded globally within this script for simplicity.

    Returns:
        scipy.sparse.csr.csr_matrix: A sparse matrix containing the transformed features.
                                    Returns None if the vectorizer failed to load.
    """
    if loaded_url_vectorizer is None:
        print("Error: URL vectorizer not loaded. Cannot extract URL features.")
        return None
    # The vectorizer expects an iterable (like a list) even for a single string
    return loaded_url_vectorizer.transform([url])

def extract_text_features(text):
    """
    Extracts numerical features from a given text using the pre-trained text vectorizer.

    Args:
        text (str): The text string to extract features from.
        # Note: The vectorizer is loaded globally within this script for simplicity.

    Returns:
        scipy.sparse.csr.csr_matrix: A sparse matrix containing the transformed features.
                                    Returns None if the vectorizer failed to load.
    """
    if loaded_text_vectorizer is None:
        print("Error: Text vectorizer not loaded. Cannot extract text features.")
        return None
    # The vectorizer expects an iterable (like a list) even for a single string
    return loaded_text_vectorizer.transform([text])

# Example Usage (optional, for testing this script directly):
if __name__ == "__main__":
    test_url = "http://www.evilphishing.com/login"
    test_text = "Click here to update your account details now!"

    print(f"\nTesting URL feature extraction for: {test_url}")
    url_features = extract_url_features(test_url)
    if url_features is not None:
        print(f"URL features shape: {url_features.shape}")

    print(f"\nTesting Text feature extraction for: {test_text}")
    text_features = extract_text_features(test_text)
    if text_features is not None:
        print(f"Text features shape: {text_features.shape}")
