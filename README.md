# SentinelPhishX: Multimodal Phishing Threat Detector

SentinelPhishX is an advanced, AI-driven phishing detection system that utilizes machine learning and fusion analysis to identify malicious URLs and emails. By analyzing text metadata alongside URL structures, the system offers high-accuracy, real-time threat intelligence to protect enterprise environments.

---

## 🚀 Features
* **Multimodal Detection Pipeline:** Combines dedicated URL and email classification models for comprehensive risk scoring.
* **Fusion Analysis Engine:** Correlates disparate threat indicators to reduce false positives.
* **Feature Extraction:** Extracts syntactic, structural, and NLP-based features from raw inputs.
* **Interactive UI:** Deployed with a clean, responsive Streamlit web interface.

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python
* **Machine Learning:** Scikit-learn, Joblib (for model serialization)
* **Data Processing:** Pandas, NumPy
* **Frontend/Deployment:** Streamlit

---

## 📂 Project Structure
```text
├── .gitignore
├── README.md
├── requirements.txt
├── dataset.csv               # Core training datasets
├── email_phishing_data.csv
├── emails.csv
├── urls.csv
├── train_model.py            # Model training & optimization script
├── email_model.py            # Inference engine for email text analysis
├── url_model.py              # Inference engine for URL string analysis
├── feature_extractor.py      # Feature engineering module
├── risk_fusion.py            # Risk scoring & correlation engine
├── text_classifier.joblib     # Pre-trained NLP models
├── text_vectorizer.joblib
├── url_classifier.joblib
└── url_vectorizer.joblib 
