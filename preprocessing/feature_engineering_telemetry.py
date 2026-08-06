import pandas as pd

# Load merged telemetry dataset
df = pd.read_csv("telemetry_merged.csv", low_memory=False)

# Min-Max Normalization
def normalize(series):
    series = pd.to_numeric(series, errors="coerce")

    if series.dropna().empty:
        return pd.Series(50, index=series.index)

    min_val = series.min()
    max_val = series.max()

    if min_val == max_val:
        return pd.Series(50, index=series.index)

    return ((series - min_val) / (max_val - min_val)) * 100


# Initialize score column
df["Infrastructure_Efficiency_Score"] = pd.NA

# NODE TELEMETRY
node_mask = df["Telemetry_Type"] == "Node"

if node_mask.any():
    cpu = normalize(df.loc[node_mask, "CPU (%)"])
    mem = normalize(df.loc[node_mask, "MEM (%)"])
    fs = normalize(df.loc[node_mask, "fs (%)"])
    energy = normalize(df.loc[node_mask, "Energy (watts)"])
    rx = normalize(df.loc[node_mask, "rx (B/sec)"])
    tx = normalize(df.loc[node_mask, "tx (B/sec)"])

    df.loc[node_mask, "Infrastructure_Efficiency_Score"] = (
        0.35 * (100 - cpu)
        + 0.25 * (100 - mem)
        + 0.15 * (100 - fs)
        + 0.15 * (100 - energy)
        + 0.05 * rx
        + 0.05 * tx
    )

# POD TELEMETRY
pod_mask = df["Telemetry_Type"] == "Pod"

if pod_mask.any():
    cpu = normalize(df.loc[pod_mask, "CPU (%)"])
    mem = normalize(df.loc[pod_mask, "MEM (B)"])
    energy = normalize(df.loc[pod_mask, "Energy (watts)"])

    df.loc[pod_mask, "Infrastructure_Efficiency_Score"] = (
        0.45 * (100 - cpu)
        + 0.35 * (100 - mem)
        + 0.20 * (100 - energy)
    )

# Round scores
df["Infrastructure_Efficiency_Score"] = (
    pd.to_numeric(df["Infrastructure_Efficiency_Score"])
    .round(2)
)

# Save
df.to_csv("telemetry_merged_with_efficiency.csv", index=False)

print("Infrastructure Efficiency Score added successfully!")
print(df[[
    "Scenario",
    "Telemetry_Type",
    "Pod_Status",
    "Infrastructure_Efficiency_Score"
]].head(10))