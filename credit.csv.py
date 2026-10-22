
import pandas as pd

# Read the original dataset on your computer
data = pd.read_csv("dataset/creditcard.csv")

# Keep a small sample, including fraud rows if the Class column exists
if "Class" in data.columns:
    fraud = data[data["Class"] == 1]
    normal = data[data["Class"] == 0].sample(
        n=min(1000, len(data[data["Class"] == 0])),
        random_state=42
    )
    demo = pd.concat([normal, fraud]).sample(frac=1, random_state=42)
else:
    demo = data.sample(n=min(1500, len(data)), random_state=42)

# Save the smaller file
demo.to_csv("dataset/demo_transactions.csv", index=False)

print("Demo dataset created!")
print("Rows:", len(demo))
print("Columns:", list(demo.columns))
