import pandas as pd

files = {
    "node_telemetry_pods_on.csv": ("Node", "Pods On"),
    "node_telemetry_pods_off.csv": ("Node", "Pods Off"),
    "pod_telemetry_pods_on.csv": ("Pod", "Pods On"),
    "pod_telemetry_pods_off.csv": ("Pod", "Pods Off"),
}

dfs = []

for filename, (telemetry_type, pod_status) in files.items():
    df = pd.read_csv(filename)

    df["Telemetry_Type"] = telemetry_type
    df["Pod_Status"] = pod_status
    df["Scenario"] = f"{telemetry_type}_{pod_status.replace(' ', '_')}"

    dfs.append(df)

# Merge while preserving every column
merged_df = pd.concat(dfs, ignore_index=True, sort=False)

# Desired column order
column_order = [
    "Scenario",
    "Telemetry_Type",
    "Pod_Status",

    # Names
    "node_name",
    "pod_name",

    "timestamp",

    # Shared
    "CPU (%)",
    "Energy (watts)",

    # Node-only
    "MEM (%)",
    "fs (%)",
    "rx (B/sec)",
    "tx (B/sec)",

    # Pod-only
    "MEM (B)"
]

# Keep existing columns in the desired order
existing = [c for c in column_order if c in merged_df.columns]

# Add any unexpected columns to the end
remaining = [c for c in merged_df.columns if c not in existing]

merged_df = merged_df[existing + remaining]

merged_df.to_csv("telemetry_merged.csv", index=False)

print("Merge successful!")
print(f"Rows: {len(merged_df)}")
print("Columns:")
print(list(merged_df.columns))