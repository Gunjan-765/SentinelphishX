import joblib # For loading the trained machine learning models
import pandas as pd # For potential data manipulation (though not heavily used in this specific script)

# --- Import Feature Extraction Functions ---
# This assumes you have a file named 'feature_extractor.py' in the same directory.
# It should contain functions to convert raw URLs and text into numerical features
# that your models understand.
try:
    from feature_extractor import extract_url_features, extract_text_features
    print("Successfully imported feature extraction functions from feature_extractor.py.")
except ImportError:
    print("Error: Could not import feature extraction functions from feature_extractor.py.")
    print("Please ensure 'feature_extractor.py' exists and contains 'extract_url_features' and 'extract_text_features'.")
    # If import fails, set these to None to prevent crashes later
    extract_url_features = None
    extract_text_features = None
    # Consider exiting here if feature extraction is critical and cannot proceed
    # exit("Exiting due to missing feature extraction functions.")


# --- Global Variables for Loaded Models and Vectorizers ---
# Initialize them to None. They will be populated if loading is successful.
url_classifier = None
text_classifier = None
url_vectorizer = None
text_vectorizer = None

# --- Load the Trained Models and Vectorizers ---
# These are the files saved by your 'train_model.py' script.
# We use try-except blocks to gracefully handle cases where files might be missing.
print("\nAttempting to load machine learning models and vectorizers...")

try:
    url_classifier = joblib.load('url_classifier.joblib')
    print("URL classifier loaded successfully.")
except FileNotFoundError:
    print("Error: 'url_classifier.joblib' not found. Please ensure you have trained and saved it using 'train_model.py'.")
except Exception as e:
    print(f"An unexpected error occurred while loading URL classifier: {e}")

try:
    text_classifier = joblib.load('text_classifier.joblib')
    print("Text classifier loaded successfully.")
except FileNotFoundError:
    print("Error: 'text_classifier.joblib' not found. Please ensure you have trained and saved it using 'train_model.py'.")
except Exception as e:
    print(f"An unexpected error occurred while loading Text classifier: {e}")

try:
    url_vectorizer = joblib.load('url_vectorizer.joblib')
    print("URL vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: 'url_vectorizer.joblib' not found. Please ensure you saved it during training in 'train_model.py'.")
except Exception as e:
    print(f"An unexpected error occurred while loading URL vectorizer: {e}")

try:
    text_vectorizer = joblib.load('text_vectorizer.joblib')
    print("Text vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: 'text_vectorizer.joblib' not found. Please ensure you saved it during training in 'train_model.py'.")
except Exception as e:
    print(f"An unexpected error occurred while loading Text vectorizer: {e}")

print("\nModel loading process completed.")

# --- Check if all critical components are loaded ---
# If any component is None, it means it failed to load, and we should stop or warn.
if all([url_classifier, text_classifier, url_vectorizer, text_vectorizer, extract_url_features, extract_text_features]):
    print("\nAll models, vectorizers, and feature extraction functions are ready for use! 🎉")
else:
    print("\nWarning: Some critical components failed to load or import. Phishing detection may not function correctly. ⚠️")
    # It's a good practice to exit if essential parts are missing, or handle gracefully
    exit("Exiting because essential models or feature extractors could not be loaded.")


# --- Core Phishing Score Prediction and Fusion Logic ---

def predict_phishing_score(url, email_text):
    """
    Predicts the overall phishing risk score for a given URL and email text.
    It uses the loaded models to get individual probabilities and then fuses them.

    Args:
        url (str): The URL string to be analyzed for phishing characteristics.
        email_text (str): The body text of the email to be analyzed for phishing characteristics.

    Returns:
        float: A fused phishing risk score, ranging from 0 to 1.
               A score closer to 1 indicates a higher probability of being phishing.
               Returns -1 if any part of the prediction process fails.
    """
    url_phish_prob = 0.0 # Initialize URL phishing probability
    text_phish_prob = 0.0 # Initialize text phishing probability

    # 1. Predict Phishing Score from URL
    # Ensure feature extraction function and vectorizer are available
    if extract_url_features and url_vectorizer:
        try:
            # Extract features using the loaded URL vectorizer
            url_features = extract_url_features(url)
            if url_features is not None:
                # Predict the probability of the URL belonging to the 'phishing' class (class 1)
                # .predict_proba() returns probabilities for [non-phishing, phishing]
                url_phish_prob = url_classifier.predict_proba(url_features)[:, 1][0]
                print(f"URL Phishing Probability: {url_phish_prob:.4f}")
            else:
                print("Could not extract features for the URL.")
                return -1 # Indicate failure
        except Exception as e:
            print(f"Error predicting URL phishing score: {e}")
            return -1 # Indicate failure
    else:
        print("URL feature extraction or vectorizer not ready.")
        return -1


    # 2. Predict Phishing Score from Text
    # Ensure feature extraction function and vectorizer are available
    if extract_text_features and text_vectorizer:
        try:
            # Extract features using the loaded text vectorizer
            text_features = extract_text_features(email_text)
            if text_features is not None:
                # Predict the probability of the text belonging to the 'phishing' class (class 1)
                text_phish_prob = text_classifier.predict_proba(text_features)[:, 1][0]
                print(f"Text Phishing Probability: {text_phish_prob:.4f}")
            else:
                print("Could not extract features for the email text.")
                return -1 # Indicate failure
        except Exception as e:
            print(f"Error predicting Text phishing score: {e}")
            return -1 # Indicate failure
    else:
        print("Text feature extraction or vectorizer not ready.")
        return -1

    # 3. Fuse the Scores
    # Currently, a simple average is used to combine the two probabilities.
    # This fusion method can be made more complex (e.g., weighted average,
    # another meta-classifier, etc.) based on your needs.
    fused_score = (url_phish_prob + text_phish_prob) / 2
    print(f"Fused Phishing Risk Score: {fused_score:.4f}")

    return fused_score


# --- Example Usage (This block runs when you execute the script directly) ---
if __name__ == "__main__":
    print("\n--- Running Example Phishing Detection Scenarios ---")

    # Example 1: A potentially legitimate email with low risk
    legit_url = "https://www.google.com/mail/inbox"
    legit_text = "Hi John, your quarterly report is attached. Please review it by end of day."
    print("\n--- Analyzing a potentially LEGITIMATE email ---")
    legit_score = predict_phishing_score(legit_url, legit_text)
    if legit_score != -1: # Check if prediction was successful
        print(f"Final Overall Risk Score: {legit_score:.4f}")
        # Determine status based on a simple threshold (e.g., 0.5)
        print(f"Status: {'Phishing Risk' if legit_score > 0.5 else 'Low Risk'}")

    # Example 2: A clearly phishing email with high risk signals
    phish_url = "http://www.badphishingsite.co/login.php?id=123"
    phish_text = "Urgent: Your account has been compromised. Click this link immediately to verify your details or your account will be suspended."
    print("\n--- Analyzing a potentially PHISHING email ---")
    phish_score = predict_phishing_score(phish_url, phish_text)
    if phish_score != -1:
        print(f"Final Overall Risk Score: {phish_score:.4f}")
        print(f"Status: {'Phishing Risk' if phish_score > 0.5 else 'Low Risk'}")

    # Example 3: Another phishing scenario, perhaps with a slightly less obvious URL
    another_phish_url = "https://paypal.com.updates.xyz/verify" # Subdomain trickery
    another_phish_text = "Your PayPal account activity looks suspicious. Please update your billing information now via the link below."
    print("\n--- Analyzing another PHISHING email example ---")
    another_phish_score = predict_phishing_score(another_phish_url, another_phish_text)
    if another_phish_score != -1:
        print(f"Final Overall Risk Score: {another_phish_score:.4f}")
        print(f"Status: {'Phishing Risk' if another_phish_score > 0.5 else 'Low Risk'}")

    # Example 4: An email with a suspicious URL but generic text (or vice-versa)
    ambiguous_url = "http://support-login-secure.biz/update"
    ambiguous_text = "Dear customer, please find attached your latest invoice." # Text might seem benign
    print("\n--- Analyzing an AMBIGUOUS email example ---")
    ambiguous_score = predict_phishing_score(ambiguous_url, ambiguous_text)
    if ambiguous_score != -1:
        print(f"Final Overall Risk Score: {ambiguous_score:.4f}")
        print(f"Status: {'Phishing Risk' if ambiguous_score > 0.5 else 'Low Risk'}")
