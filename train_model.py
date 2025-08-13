# train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ==== File paths ====
DATA_FILE = "data/cleaned_news.csv"
MODEL_FILE = "models/fake_news_model.joblib"
VECTORIZER_FILE = "models/tfidf_vectorizer.joblib"

# ==== 1. Load dataset ====
if not os.path.exists(DATA_FILE):
    raise SystemExit(f"Dataset not found: {DATA_FILE}. Run scripts/update_dataset.py first.")

print(f"Loading dataset from {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

# ==== 2. Clean & normalize labels ====
df = df.dropna(subset=["content", "label"])
df["label"] = df["label"].astype(str).str.strip().str.lower()

# Map textual labels to numeric
label_map = {
    "real": 0,
    "fake": 1,
    "0": 0,
    "1": 1
}
df["label"] = df["label"].map(label_map)

# Drop rows with unmapped labels
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print(f"Dataset cleaned. Shape: {df.shape}")
print(f"Label counts:\n{df['label'].value_counts()}")

# ==== 3. Features and labels ====
X = df["content"]
y = df["label"]

# ==== 4. Train-test split ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==== 5. TF-IDF Vectorization ====
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==== 6. Train Logistic Regression ====
print("Training model...")
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",  # helps with imbalance
    solver="liblinear"
)
model.fit(X_train_tfidf, y_train)

# ==== 7. Predictions ====
y_pred = model.predict(X_test_tfidf)

# ==== 8. Evaluation ====
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==== 9. Save model & vectorizer ====
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_FILE)
joblib.dump(vectorizer, VECTORIZER_FILE)
print(f"Model saved to {MODEL_FILE}")
print(f"Vectorizer saved to {VECTORIZER_FILE}")
