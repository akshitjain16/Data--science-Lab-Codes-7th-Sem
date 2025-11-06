# Task 4: Visualization of Sensor Data (Using Matplotlib + Seaborn)
# ---------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ---------------------------------------------------------------
# 1️⃣ Load the CSV files produced by the pipeline
# ---------------------------------------------------------------
agg_file = "output_pipeline/rolling_aggregates.csv"
alert_file = "output_pipeline/anomaly_alerts.csv"

try:
    df_agg = pd.read_csv(agg_file)
    df_alert = pd.read_csv(alert_file)
    print("✅ Files loaded successfully.")
except FileNotFoundError:
    print("❌ Required CSV files not found. Please run the pipeline first.")
    exit()

# Rename timestamp column if present
if "ts" in df_agg.columns:
    df_agg["ts"] = pd.to_datetime(df_agg["ts"])

if "ts" in df_alert.columns:
    df_alert["ts"] = pd.to_datetime(df_alert["ts"])

# ---------------------------------------------------------------
# 2️⃣ Plot average temperature and humidity over time
# ---------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df_agg["ts"], df_agg["avg_temp"], label="Average Temperature (°C)")
plt.plot(df_agg["ts"], df_agg["avg_humidity"], label="Average Humidity (%)")
plt.title("Average Temperature & Humidity Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Values")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 3️⃣ Seaborn barplot for sensor-wise averages
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x="sensor_id", y="avg_temp", data=df_agg, color="orange")
plt.title("Sensor-wise Average Temperature")
plt.xlabel("Sensor ID")
plt.ylabel("Average Temperature (°C)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="sensor_id", y="avg_humidity", data=df_agg, color="skyblue")
plt.title("Sensor-wise Average Humidity")
plt.xlabel("Sensor ID")
plt.ylabel("Average Humidity (%)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 4️⃣ Scatter plot for anomalies
# ---------------------------------------------------------------
if not df_alert.empty:
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x="ts", y="value", hue="sensor_id",
                    data=df_alert[df_alert["metric"] == "temperature"],
                    palette="tab10", s=100, edgecolor="k")
    plt.title("Temperature Anomalies Detected")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    plt.show()
else:
    print("ℹ️ No anomalies recorded.")

# ---------------------------------------------------------------
# 5️⃣ Correlation heatmap (only valid for numeric columns)
# ---------------------------------------------------------------
numeric_cols = ["avg_temp", "avg_humidity", "count"]
plt.figure(figsize=(6, 4))
sns.heatmap(df_agg[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
