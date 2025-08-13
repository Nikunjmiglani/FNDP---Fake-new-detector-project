import streamlit as st
import joblib
import re

# ---------------------------
# Load Model & Vectorizer
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/fake_news_model.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_artifacts()

# ---------------------------
# Text Cleaning (must match training)
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# ---------------------------
# Detect gibberish input
# ---------------------------
def is_gibberish(text):
    if re.match(r'http[s]?://', text):  # Allow URLs
        return False
    if len(text.split()) < 3:           # Too short
        return True
    if len(set(text)) < 5:              # Too few unique chars
        return True
    return False

# ---------------------------
# Sensitive Claim Detection
# ---------------------------
def contains_sensitive_claim(text):
    sensitive_keywords = [
        "dies", "dead", "death", "killed", "murder", "assassinated",
        "heart attack", "suicide", "bomb", "arrested", "hospitalized", "rape"
    ]

    public_figures = [
        "trump", "biden", "modi", "putin", "zelensky", "elon musk", "obama",
        "kamala", "xi jinping", "pope", "queen", "kardashian", "nadal"
    ]

    text = text.lower()
    for word in sensitive_keywords + public_figures:
        if word in text:
            return True
    return False

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter a **news headline** or **full article** to check if it's **Real** or **Fake**.")

user_input = st.text_area("Paste your news here:")

if st.button("Check News"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to check.")
    elif is_gibberish(user_input):
        st.error("🚨 This news seems **FAKE** due to nonsensical or short input.")
    elif contains_sensitive_claim(user_input):
        st.warning("⚠️ This input contains **sensitive content or public figure claims**.\n\nAI predictions may be unreliable. Please verify using trusted news sources.")
    else:
        # Preprocess and Predict
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        proba = model.predict_proba(vectorized_input)[0]

        fake_score = round(proba[1] * 100, 2)  # % fake
        real_score = round(proba[0] * 100, 2)  # % real

        if real_score >= fake_score:
            st.success(f"✅ This news seems **REAL**.\n\n🧾 Confidence: {real_score}% real, {fake_score}% fake")
        else:
            st.error(f"🚨 This news is likely **FAKE**.\n\n🧾 Confidence: {fake_score}% fake, {real_score}% real")

# Footer Disclaimer
st.markdown("---")
st.caption("🔍 Disclaimer: AI predictions are not substitutes for professional fact-checking. Always verify critical claims through trusted news organizations.")
