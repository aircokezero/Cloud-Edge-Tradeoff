import pandas as pd

# ==========================
# CONFIGURATION
# ==========================

FILES = {
    "IIoT Dataset": "iiot_with_efficiency.csv",
    "Telemetry Dataset": "telemetry_merged_with_efficiency.csv"
}

# ==========================
# VALIDATION FUNCTION
# ==========================

def validate_dataset(name, filepath):

    print("=" * 70)
    print(f"VALIDATING: {name}")
    print("=" * 70)

    df = pd.read_csv(filepath, low_memory=False)

    # ----------------------
    # Shape
    # ----------------------
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # ----------------------
    # Missing Values
    # ----------------------
    print("\nMissing Values:")

    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("No missing values")
    else:
        print(missing[missing > 0])

    # ----------------------
    # Duplicate Rows
    # ----------------------
    duplicates = df.duplicated().sum()

    print(f"\nDuplicate Rows: {duplicates}")

    # ----------------------
    # Data Types
    # ----------------------
    print("\nColumn Types:")

    print(df.dtypes)

    # ----------------------
    # Timestamp Check
    # ----------------------
    if "timestamp" in df.columns:

        invalid = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        ).isna().sum()

        print(f"\nInvalid timestamps: {invalid}")

    if "Timestamp" in df.columns:

        invalid = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        ).isna().sum()

        print(f"\nInvalid timestamps: {invalid}")

    # ----------------------
    # Efficiency Score
    # ----------------------
    for col in [
        "Edge_Efficiency_Score",
        "Infrastructure_Efficiency_Score",
        "Cloud_Efficiency_Score"
    ]:

        if col in df.columns:

            invalid = ((df[col] < 0) | (df[col] > 100)).sum()

            print(f"\n{col}")

            print(f"Minimum : {df[col].min():.2f}")
            print(f"Maximum : {df[col].max():.2f}")
            print(f"Mean    : {df[col].mean():.2f}")
            print(f"Outside 0-100 : {invalid}")

    # ----------------------
    # Numeric Summary
    # ----------------------
    print("\nNumeric Summary:")

    print(df.describe())

    # ----------------------
    # Category Validation
    # ----------------------
    if "Maintenance_Status" in df.columns:

        expected = {
            "Normal",
            "Warning",
            "Critical"
        }

        actual = set(df["Maintenance_Status"].dropna().unique())

        invalid = actual - expected

        print("\nMaintenance Status")

        if len(invalid) == 0:
            print("✔ Valid")
        else:
            print("Unexpected values:", invalid)

    if "Telemetry_Type" in df.columns:

        expected = {"Node", "Pod"}

        actual = set(df["Telemetry_Type"].dropna().unique())

        invalid = actual - expected

        print("\nTelemetry Type")

        if len(invalid) == 0:
            print("Valid")
        else:
            print("Unexpected values:", invalid)

    print("\nValidation Complete.")
    print()


# ==========================
# RUN
# ==========================

for dataset, file in FILES.items():
    validate_dataset(dataset, file)