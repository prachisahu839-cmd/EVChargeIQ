import json
import pandas as pd

# Load raw ACN data
with open("data/raw/caltech_sessions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("\n========== BASIC INFORMATION ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print("Duplicate sessionID:", df["sessionID"].duplicated().sum())

print("\n========== DATE RANGE ==========")
df["connectionTime"] = pd.to_datetime(df["connectionTime"], errors="coerce")

print("First connection:", df["connectionTime"].min())
print("Last connection:", df["connectionTime"].max())

print("\n========== ENERGY ==========")
print(df["kWhDelivered"].describe())

print("\n========== UNIQUE VALUES ==========")
print("Stations:", df["stationID"].nunique())
print("Spaces:", df["spaceID"].nunique())
print("Users:", df["userID"].nunique())
print("Sites:", df["siteID"].nunique())

print("\n========== SAMPLE ==========")
print(df.head(3).to_string())