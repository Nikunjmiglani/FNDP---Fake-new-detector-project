import pandas as pd

# Load CSV files from data folder
fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")

print("Fake news shape:", fake_df.shape)
print("Real news shape:", real_df.shape)

print("\nFake news columns:", fake_df.columns.tolist())
print("Real news columns:", real_df.columns.tolist())

# See first 2 rows from each
print("\n--- Fake news sample ---")
print(fake_df.head(2))

print("\n--- Real news sample ---")
print(real_df.head(2))
