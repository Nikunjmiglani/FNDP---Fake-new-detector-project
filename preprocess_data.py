import pandas as pd
import re

# Load datasets
fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")

# Add labels (1 = fake, 0 = real)
fake_df["label"] = 1
real_df["label"] = 0

# Combine into one dataframe
df = pd.concat([fake_df, real_df], ignore_index=True)

# Function to clean text
def clean_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'http\S+', ' ', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)  # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# Combine title + text into content
df["content"] = (df["title"].fillna('') + " " + df["text"].fillna('')).apply(clean_text)

# Drop unused columns
df = df[["content", "label"]]

# Save cleaned dataset
df.to_csv("data/cleaned_news.csv", index=False)

print("Cleaning complete! Saved to data/cleaned_news.csv")
print(df.head())
