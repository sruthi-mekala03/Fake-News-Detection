import pandas as pd

print("Reading datasets...")

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print("Merging datasets...")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

data.to_csv("data/news.csv", index=False)

print("news.csv created successfully!")