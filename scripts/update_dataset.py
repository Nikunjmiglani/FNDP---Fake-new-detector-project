import os
import pandas as pd
import re
from datetime import datetime

DATA_DIR = "data"
EXISTING = os.path.join(DATA_DIR, "cleaned_news.csv")
OUT = EXISTING
BACKUP = os.path.join(DATA_DIR, f"cleaned_news_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
NEW_FILES = [
    os.path.join(DATA_DIR, "new_real_newsapi.csv"),
    os.path.join(DATA_DIR, "new_real_rss.csv"),
    # add other new files if you collect fake ones
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)   # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load existing dataset or create from Fake/True
if os.path.exists(EXISTING):
    df_existing = pd.read_csv(EXISTING)
else:
    f_fake = os.path.join(DATA_DIR, "Fake.csv")
    f_true = os.path.join(DATA_DIR, "True.csv")
    if os.path.exists(f_fake) and os.path.exists(f_true):
        df_fake = pd.read_csv(f_fake)
        df_true = pd.read_csv(f_true)
        df_fake["label"] = 1
        df_true["label"] = 0
        df_existing = pd.concat([df_fake, df_true], ignore_index=True)
        if "title" in df_existing.columns and "text" in df_existing.columns:
            df_existing["content"] = (df_existing["title"].fillna('') + ' ' + df_existing["text"].fillna(''))
        elif "content" not in df_existing.columns and "text" in df_existing.columns:
            df_existing["content"] = df_existing["text"]
    else:
        df_existing = pd.DataFrame(columns=["content", "label"])

# Load new files
dfs = [df_existing]
for f in NEW_FILES:
    if os.path.exists(f):
        df = pd.read_csv(f)
        if "content" not in df.columns and "title" in df.columns:
            df["content"] = df["title"].fillna('') + " " + df.get("content", "").fillna('')
        if "label" not in df.columns:
            df["label"] = 0
        dfs.append(df[["content", "label"]])

combined = pd.concat(dfs, ignore_index=True)

# Clean text
combined["content"] = combined["content"].fillna("").map(clean_text)
combined = combined[combined["content"].str.len() > 20].copy()

# Drop duplicates
before = len(combined)
combined = combined.drop_duplicates(subset=["content"])
after = len(combined)

# Backup old cleaned file
if os.path.exists(EXISTING):
    os.rename(EXISTING, BACKUP)
    print(f"Backed up old cleaned file to {BACKUP}")

combined.to_csv(OUT, index=False, encoding="utf-8")
print(f"Updated cleaned dataset saved to {OUT} ({before}->{after} after dedupe).")
print(combined["label"].value_counts())
