"""
Real-Time Sensor Data Pipeline (Pure Python)
- Simulates IoT sensor readings (temperature, humidity)
- Streams via a Queue (as if it were Kafka)
- Processes rolling 1-minute aggregates
- Simple anomaly detection (z-score + thresholds)
- Logs results and saves to CSV periodically
- Optional live plotting (toggle LIVE_PLOT)
"""

import time
import json
import random
import threading
import queue
from dataclasses import dataclass, asdict
from typing import Dict, List
import signal
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
SENSORS = ["sensor_1", "sensor_2", "sensor_3"]
PRODUCE_INTERVAL_SEC = 1.0          # how often each reading is produced
ROLLING_WINDOW_SEC = 60             # aggregation window (1 minute)
SAVE_EVERY_N_RECORDS = 50           # write to CSV every N messages processed
OUTPUT_DIR = "output_pipeline"
RAW_CSV_PATH = os.path.join(OUTPUT_DIR, "raw_stream.csv")
AGG_CSV_PATH = os.path.join(OUTPUT_DIR, "rolling_aggregates.csv")
ALERTS_CSV_PATH = os.path.join(OUTPUT_DIR, "anomaly_alerts.csv")
LIVE_PLOT = False                    # set False if you don’t want live chart

random.seed(42)
np.random.seed(42)


# ----------------------------
# Setup helpers
# ----------------------------
def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def init_csv_headers() -> None:
    """Create CSVs with headers if files don't exist (no-op if present)."""
    if not os.path.exists(RAW_CSV_PATH):
        pd.DataFrame(columns=["ts", "sensor_id", "temperature", "humidity"]).to_csv(
            RAW_CSV_PATH, index=False
        )
    if not os.path.exists(AGG_CSV_PATH):
        pd.DataFrame(columns=["ts", "sensor_id", "avg_temp", "avg_humidity", "count"]).to_csv(
            AGG_CSV_PATH, index=False
        )
    if not os.path.exists(ALERTS_CSV_PATH):
        pd.DataFrame(columns=["ts", "sensor_id", "metric", "value", "reason"]).to_csv(
            ALERTS_CSV_PATH, index=False
        )


def setup_signal_handler(stop_evt: threading.Event) -> None:
    def handle_sigint(sig, frame):
        print("\n🛑 Stopping…")
        stop_evt.set()
    signal.signal(signal.SIGINT, handle_sigint)


# ----------------------------
# Data Model
# ----------------------------
@dataclass
class SensorReading:
    sensor_id: str
    temperature: float  # °C
    humidity: float     # %
    ts: float           # epoch seconds

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# ----------------------------
# Producer: simulates sensors
# ----------------------------
class SensorProducer(threading.Thread):
    def __init__(self, q: queue.Queue, stop_evt: threading.Event):
        super().__init__(daemon=True)
        self.q = q
        self.stop_evt = stop_evt

    def run(self):
        print("▶️ Producer started.")
        while not self.stop_evt.is_set():
            reading = SensorReading(
                sensor_id=random.choice(SENSORS),
                temperature=round(random.uniform(20.0, 40.0), 2),
                humidity=round(random.uniform(30.0, 70.0), 2),
                ts=time.time()
            )
            self.q.put(reading)
            # print(f"[PRODUCED] {reading}")
            time.sleep(PRODUCE_INTERVAL_SEC)
        print("⏹️ Producer stopped.")


# ----------------------------
# Processor: consumes, aggregates, detects anomalies
# ----------------------------
class DataProcessor(threading.Thread):
    def __init__(self, q: queue.Queue, stop_evt: threading.Event):
        super().__init__(daemon=True)
        self.q = q
        self.stop_evt = stop_evt
        self.buffer: List[SensorReading] = []
        self.processed_count = 0

        # Live plot setup
        self.fig = None
        self.ax = None
        if LIVE_PLOT:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(9, 4))
            self.ax.set_title("Live Avg Temperature (last 60s) by Sensor")
            self.ax.set_xlabel("Sensor")
            self.ax.set_ylabel("Avg Temp (°C)")

        # Ensure CSVs exist with headers
        init_csv_headers()

    def run(self):
        print("▶️ Processor started.")
        while not self.stop_evt.is_set():
            try:
                reading: SensorReading = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            self.buffer.append(reading)
            self.processed_count += 1

            # Append raw to CSV (appending in small batches keeps it simple)
            pd.DataFrame([{
                "ts": reading.ts,
                "sensor_id": reading.sensor_id,
                "temperature": reading.temperature,
                "humidity": reading.humidity
            }]).to_csv(RAW_CSV_PATH, mode="a", header=False, index=False)

            # Drop old rows outside rolling window
            cutoff = time.time() - ROLLING_WINDOW_SEC
            self.buffer = [r for r in self.buffer if r.ts >= cutoff]

            # Compute aggregates on the last 60s per sensor
            df = pd.DataFrame([asdict(r) for r in self.buffer])
            if not df.empty:
                agg = df.groupby("sensor_id").agg(
                    avg_temp=("temperature", "mean"),
                    avg_humidity=("humidity", "mean"),
                    count=("temperature", "count")
                ).reset_index()
                agg["ts"] = time.time()

                # Save aggregate snapshot every N processed messages
                if self.processed_count % SAVE_EVERY_N_RECORDS == 0:
                    agg[["ts", "sensor_id", "avg_temp", "avg_humidity", "count"]].to_csv(
                        AGG_CSV_PATH, mode="a", header=False, index=False
                    )

                # Detect anomalies (simple rules + z-score)
                self.detect_anomalies(df)

                # Update live plot
                if LIVE_PLOT:
                    self.update_plot(agg)

        print("⏹️ Processor stopped.")

    def detect_anomalies(self, df: pd.DataFrame):
        # Rule-based thresholds
        thresh_alerts = []
        over_temp = df[df["temperature"] > 38.0]
        under_temp = df[df["temperature"] < 21.0]
        over_hum = df[df["humidity"] > 68.0]
        under_hum = df[df["humidity"] < 32.0]

        for _, r in over_temp.iterrows():
            thresh_alerts.append({"ts": r["ts"], "sensor_id": r["sensor_id"], "metric": "temperature",
                                  "value": r["temperature"], "reason": "temp>38°C"})
        for _, r in under_temp.iterrows():
            thresh_alerts.append({"ts": r["ts"], "sensor_id": r["sensor_id"], "metric": "temperature",
                                  "value": r["temperature"], "reason": "temp<21°C"})
        for _, r in over_hum.iterrows():
            thresh_alerts.append({"ts": r["ts"], "sensor_id": r["sensor_id"], "metric": "humidity",
                                  "value": r["humidity"], "reason": "hum>68%"})
        for _, r in under_hum.iterrows():
            thresh_alerts.append({"ts": r["ts"], "sensor_id": r["sensor_id"], "metric": "humidity",
                                  "value": r["humidity"], "reason": "hum<32%"})

        # Z-score anomaly (per sensor, last 60s)
        z_alerts = []
        for sid, g in df.groupby("sensor_id"):
            if len(g) >= 10:
                for metric in ["temperature", "humidity"]:
                    mu = g[metric].mean()
                    sigma = g[metric].std(ddof=0)
                    if sigma == 0:
                        continue
                    z = (g[metric] - mu) / sigma
                    outliers = g[np.abs(z) > 2.5]
                    for _, r in outliers.iterrows():
                        z_alerts.append({"ts": r["ts"], "sensor_id": sid, "metric": metric,
                                         "value": r[metric], "reason": f"z>|2.5| ({metric})"})

        alerts = thresh_alerts + z_alerts
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            alerts_df.to_csv(ALERTS_CSV_PATH, mode="a", header=False, index=False)
            # Print a compact console summary
            for a in alerts[:3]:
                t = time.strftime("%H:%M:%S", time.localtime(a["ts"]))
                print(f"⚠️  {t}  {a['sensor_id']}  {a['metric']}={a['value']}  ({a['reason']})")
            if len(alerts) > 3:
                print(f"… and {len(alerts)-3} more alerts (written to {ALERTS_CSV_PATH}).")

    def update_plot(self, agg: pd.DataFrame):
        self.ax.clear()
        self.ax.set_title("Live Avg Temperature (last 60s) by Sensor")
        self.ax.set_xlabel("Sensor")
        self.ax.set_ylabel("Avg Temp (°C)")

        # Ensure sensors with no rows still show up
        plot_df = pd.DataFrame({"sensor_id": SENSORS}).merge(
            agg[["sensor_id", "avg_temp"]], on="sensor_id", how="left"
        )
        plot_df["avg_temp"] = plot_df["avg_temp"].fillna(0.0)

        self.ax.bar(plot_df["sensor_id"], plot_df["avg_temp"])
        self.ax.set_ylim(0, 45)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# ----------------------------
# Graceful shutdown / Orchestration
# ----------------------------
def run_pipeline():
    ensure_output_dir()
    init_csv_headers()

    stop_evt = threading.Event()
    q: "queue.Queue[SensorReading]" = queue.Queue(maxsize=1000)

    setup_signal_handler(stop_evt)

    prod = SensorProducer(q, stop_evt)
    proc = DataProcessor(q, stop_evt)

    prod.start()
    proc.start()

    # Wait until both threads stop
    while prod.is_alive() or proc.is_alive():
        time.sleep(0.2)
        if stop_evt.is_set():
            break

    print(f"\n📁 Files written:\n- {RAW_CSV_PATH}\n- {AGG_CSV_PATH}\n- {ALERTS_CSV_PATH}")
    if LIVE_PLOT:
        try:
            plt.ioff()
            plt.show(block=False)
        except Exception:
            pass
    print("✅ Pipeline ended cleanly.")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
