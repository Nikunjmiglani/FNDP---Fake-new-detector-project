import sys
import os
import pandas as pd
import re
from datetime import datetime

# Force UTF-8 for Windows console output
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data"
EXISTING = os.path.join(DATA_DIR, "cleaned_news.csv")
OUT = EXISTING
BACKUP = os.path.join(DATA_DIR, f"cleaned_news_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

def clean_text(text):
    """Lowercase, remove URLs, keep only letters/spaces, normalize spaces."""
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)  # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load existing cleaned dataset or create empty DataFrame
if os.path.exists(EXISTING):
    df_existing = pd.read_csv(EXISTING, on_bad_lines="skip")
else:
    df_existing = pd.DataFrame(columns=["content", "label"])

# Gather all CSV files except backups & cleaned file
all_csv_files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".csv")
    and "backup" not in f.lower()
    and f != "cleaned_news.csv"
]

dfs = [df_existing]

for file in all_csv_files:
    try:
        df = pd.read_csv(file, on_bad_lines="skip")

        # Combine title + content if content missing
        if "content" not in df.columns:
            if "title" in df.columns:
                df["content"] = df["title"].fillna('') + " " + df.get("text", "").fillna('')
            elif "text" in df.columns:
                df["content"] = df["text"].fillna('')
            else:
                print(f"Skipping {file} — no content, title, or text column.")
                continue

        # Assign default label if missing
        if "label" not in df.columns:
            df["label"] = 1 if "fake" in file.lower() else 0

        dfs.append(df[["content", "label"]])
        print(f"Loaded {len(df)} rows from {os.path.basename(file)}")

    except Exception as e:
        print(f"Error reading {os.path.basename(file)}: {e}")

# Combine all datasets
combined = pd.concat(dfs, ignore_index=True)

# Clean text
combined["content"] = combined["content"].fillna("").map(clean_text)

# Remove very short entries
combined = combined[combined["content"].str.len() > 20].copy()

# Remove duplicates
before = len(combined)
combined = combined.drop_duplicates(subset=["content"])
after = len(combined)

# Backup old cleaned file
if os.path.exists(EXISTING):
    os.rename(EXISTING, BACKUP)
    print(f"Backed up old cleaned file to {os.path.basename(BACKUP)}")

# Save cleaned dataset
combined.to_csv(OUT, index=False, encoding="utf-8")
print(f"Updated cleaned dataset saved to {os.path.basename(OUT)} ({before} → {after} after dedupe)")
print("Label counts:")
print(combined["label"].value_counts())
