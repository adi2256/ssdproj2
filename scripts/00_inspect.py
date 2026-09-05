import os
import pandas as pd

# Location of the raw Alibaba SSD dataset
RAW_DIR = "data/raw"

print("Checking Alibaba SSD dataset...\n")

# Show the folders/files inside data/raw
items = os.listdir(RAW_DIR)

print("Items found in data/raw:")
for item in items:
    print(" -", item)

print("\nChecking one SMART CSV file...")

# Find the first 2018 CSV file
year_folder = os.path.join(RAW_DIR, "smartlog2018ssd")

csv_files = [
    f for f in os.listdir(year_folder)
    if f.endswith(".csv")
]

print("Number of CSV files found:", len(csv_files))

if csv_files:
    first_file = os.path.join(year_folder, csv_files[0])

    print("Reading:", csv_files[0])

    # IMPORTANT: only read a small sample
    df = pd.read_csv(first_file, nrows=1000)

    print("\nDataset shape of sample:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values in sample:")
    print(df.isnull().sum())
else:
    print("No CSV files found.")