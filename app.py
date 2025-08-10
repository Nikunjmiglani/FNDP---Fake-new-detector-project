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

# Function to check for gibberish or overly repetitive input
def is_gibberish(text):
    # If the text is a URL or has a URL pattern, we don't want to flag it
    if re.match(r'http[s]?://', text):
        return False  # Allow URLs to be passed through
    # If the text is too short or if it has a low variety of characters, treat as gibberish
    if len(text.split()) < 3:  # Too short, likely not meaningful
        return True
    if len(set(text)) < 5:  # Too few unique characters, likely gibberish
        return True
    return False

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
        # Check for gibberish or nonsensical input
        if is_gibberish(user_input):
            st.error("🚨 This news seems **FAKE** due to nonsensical input.")
        else:
            # Preprocess and predict
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            proba = model.predict_proba(vectorized_input)[0]

            fake_score = proba[1]  # Probability it's fake
            real_score = proba[0]  # Probability it's real

            # Display the confidence scores for debugging
            st.write(f"Real Score: {real_score:.2f}, Fake Score: {fake_score:.2f}")

            # Default threshold for fake news
            threshold = 0.5  # Set a stricter threshold to catch more fake news

            # Adjust threshold for India-related keywords
            india_keywords = ["india", "delhi", "mumbai", "kolkata", "bengaluru", "chennai"]
            if any(word in cleaned_input.lower() for word in india_keywords):
                threshold -= 0.2  # Loosen threshold for India-related news

            # Specific fake-related keywords or patterns
            attack_keywords = ["attack", "war", "invasion", "military strike"]
            if any(word in cleaned_input.lower() for word in attack_keywords):
                threshold = 0.3  # Tighten the threshold for extreme claims

            # Decision making based on stricter threshold
            if fake_score > threshold:
                st.error("🚨 This news is likely **FAKE**.")
            else:
                st.success("✅ This news seems **REAL**.")
