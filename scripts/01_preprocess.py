import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

SMART_2018 = os.path.join(RAW_DIR, "smartlog2018ssd")
SMART_2019 = os.path.join(RAW_DIR, "smartlog2019ssd")

FAILURE_DIR = os.path.join(RAW_DIR, "ssd_failure_label.csv")

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "alibaba_ssd_processed.parquet"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND FAILURE LABEL FILE
# ============================================================

failure_files = glob.glob(
    os.path.join(FAILURE_DIR, "**", "*.csv"),
    recursive=True
)

if not failure_files:
    raise FileNotFoundError(
        "Could not find ssd_failure_label.csv"
    )

FAILURE_FILE = failure_files[0]

print("Failure label file:")
print(FAILURE_FILE)


# ============================================================
# LOAD FAILURE LABELS
# ============================================================

print("\nLoading failure labels...")

failure_df = pd.read_csv(FAILURE_FILE)

print("Failure label shape:", failure_df.shape)

print("\nFailure label columns:")
print(failure_df.columns.tolist())


# Convert failure time to datetime
failure_df["failure_time"] = pd.to_datetime(
    failure_df["failure_time"],
    errors="coerce"
)


# Convert disk_id to integer
failure_df["disk_id"] = pd.to_numeric(
    failure_df["disk_id"],
    errors="coerce"
)


# Remove invalid labels
failure_df = failure_df.dropna(
    subset=["disk_id", "failure_time"]
)


failure_df["disk_id"] = failure_df["disk_id"].astype("int64")


# Keep only required columns
failure_df = failure_df[
    ["model", "disk_id", "failure_time"]
]


# Make model type consistent
failure_df["model"] = failure_df["model"].astype("string")


print(
    "Valid failure labels:",
    len(failure_df)
)


# ============================================================
# FIND SMART CSV FILES
# ============================================================

smart_files = []

for folder in [SMART_2018, SMART_2019]:

    files = sorted(
        glob.glob(
            os.path.join(folder, "*.csv")
        )
    )

    print(
        f"\nFound {len(files)} SMART files in {folder}"
    )

    smart_files.extend(files)


print(
    "\nTotal SMART CSV files:",
    len(smart_files)
)


if not smart_files:
    raise FileNotFoundError(
        "No SMART CSV files were found."
    )


# ============================================================
# PROCESS FILES ONE AT A TIME
# ============================================================

writer = None
total_rows = 0

print("\nStarting SMART data processing...\n")


for file_number, file_path in enumerate(
    smart_files,
    start=1
):

    filename = os.path.basename(file_path)

    print(
        f"[{file_number}/{len(smart_files)}] "
        f"Processing {filename}"
    )


    # --------------------------------------------------------
    # READ ONE DAILY FILE
    # --------------------------------------------------------

    df = pd.read_csv(file_path)


    # --------------------------------------------------------
    # CONVERT DATE
    # --------------------------------------------------------

    df["ds"] = pd.to_datetime(
        df["ds"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )


    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    df = df.dropna(
        subset=["disk_id", "ds"]
    )


    # --------------------------------------------------------
    # CONVERT DISK ID
    # --------------------------------------------------------

    df["disk_id"] = pd.to_numeric(
        df["disk_id"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["disk_id"]
    )

    df["disk_id"] = df["disk_id"].astype("int64")


    # --------------------------------------------------------
    # CONVERT MODEL
    # --------------------------------------------------------

    df["model"] = df["model"].astype("string")


    # --------------------------------------------------------
    # CONVERT SMART FEATURES TO FLOAT
    # --------------------------------------------------------

    metadata_columns = [
        "disk_id",
        "ds",
        "model"
    ]

    smart_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    for column in smart_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).astype("float64")


    # --------------------------------------------------------
    # ATTACH FAILURE INFORMATION
    # --------------------------------------------------------

    df = df.merge(
        failure_df,
        on=["model", "disk_id"],
        how="left"
    )


    # --------------------------------------------------------
    # CALCULATE DAYS UNTIL FAILURE
    # --------------------------------------------------------

    df["days_to_failure"] = (
        df["failure_time"] - df["ds"]
    ).dt.total_seconds() / 86400

    df["days_to_failure"] = pd.to_numeric(
        df["days_to_failure"],
        errors="coerce"
    ).astype("float64")


    # --------------------------------------------------------
    # FAILURE EVENT
    #
    # 1 = observation is on or after failure date
    # 0 = otherwise
    # --------------------------------------------------------

    df["failure_event"] = (
        df["days_to_failure"] <= 0
    ).astype("int8")


    # --------------------------------------------------------
    # ENSURE FAILURE TIME TYPE
    # --------------------------------------------------------

    df["failure_time"] = pd.to_datetime(
        df["failure_time"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # CREATE PYARROW TABLE
    # --------------------------------------------------------

    table = pa.Table.from_pandas(
        df,
        preserve_index=False
    )


    # --------------------------------------------------------
    # CREATE PARQUET WRITER
    # --------------------------------------------------------

    if writer is None:

        writer = pq.ParquetWriter(
            OUTPUT_FILE,
            table.schema,
            compression="snappy"
        )

    else:

        # Force every file to use exactly
        # the same schema as the first file

        table = table.cast(
            writer.schema,
            safe=False
        )


    # --------------------------------------------------------
    # WRITE DATA
    # --------------------------------------------------------

    writer.write_table(table)

    total_rows += len(df)

    print(
        f"    Rows processed: {len(df):,}"
    )


# ============================================================
# CLOSE PARQUET WRITER
# ============================================================

if writer is not None:
    writer.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PHASE 1 PREPROCESSING COMPLETE")
print("=" * 60)

print(
    f"Total SMART rows processed: {total_rows:,}"
)

print(
    f"Processed dataset saved to:\n{OUTPUT_FILE}"
)

print("\nDataset contains:")
print(" - SSD identifier: disk_id")
print(" - Date: ds")
print(" - SSD model: model")
print(" - SMART attributes")
print(" - Failure date: failure_time")
print(" - Days until failure: days_to_failure")
print(" - Failure event: failure_event")