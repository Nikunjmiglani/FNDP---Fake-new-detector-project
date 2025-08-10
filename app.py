import streamlit as st
import joblib
import re

# Load model & vectorizer
model = joblib.load("models/fake_news_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# Function to clean text (match training preprocessing)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation & numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

# Streamlit app UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Enter a news headline or article below to check if it's **Real** or **Fake**.")

# User input
user_input = st.text_area("Paste your news here:")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to check.")
    else:
        # Preprocess and predict
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        proba = model.predict_proba(vectorized_input)[0]

        fake_score = proba[1]  # Probability it's fake
        real_score = proba[0]  # Probability it's real

        # Default threshold for fake news
        threshold = 0.8  # Set a stricter threshold to catch more fake news

        # Adjust threshold for India-related keywords
        india_keywords = ["india", "delhi", "mumbai", "kolkata", "bengaluru", "chennai"]
        if any(word in cleaned_input.lower() for word in india_keywords):
            threshold -= 0.1  # Loosen threshold for India-related news

        # Adjust further for specific "fake news" patterns (optional)
        fake_keywords = ["scam", "conspiracy", "fake", "hoax", "fraud"]
        if any(word in cleaned_input.lower() for word in fake_keywords):
            threshold = 0.9  # Tighten the threshold if "fake" keywords are found

        # Decision making based on stricter threshold
        if fake_score > threshold:
            st.error("🚨 This news is likely **FAKE**.")
        else:
            st.success("✅ This news seems **REAL**.")
