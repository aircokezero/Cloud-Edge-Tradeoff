import pandas as pd

df = pd.read_csv("iiot_edge_computing_dataset.csv")

def normalize(col):
    return (col - col.min()) / (col.max() - col.min()) * 100

# Normalize features
df["Latency_Norm"] = normalize(df["Network_Latency"])
df["EdgeTime_Norm"] = normalize(df["Edge_Processing_Time"])
df["Vibration_Norm"] = normalize(df["Vibration"])
df["Temperature_Norm"] = normalize(df["Temperature"])

# Create efficiency score
df["Edge_Efficiency_Score"] = (
    0.35 * (100 - df["Latency_Norm"])
    + 0.35 * (100 - df["EdgeTime_Norm"])
    + 0.15 * (100 - df["Vibration_Norm"])
    + 0.10 * (100 - df["Temperature_Norm"])
    + 0.05 * (100 - df["Predicted_Failure"] * 100)
)

df.to_csv("iiot_with_efficiency.csv", index=False)

print(df[["Edge_Efficiency_Score"]].head())