import pandas as pd

from src.ingestion import fetch_metric_series
from src.logging_config import configure_logging

DAYS = 10

configure_logging()
s = fetch_metric_series("node_memory_MemAvailable_bytes", minutes=DAYS * 24 * 60)

if s.empty:
    raise SystemExit("no data in range")

print(f"\ntotal points: {len(s)}")
print(f"first: {s.index[0]}")
print(f"last:  {s.index[-1]}")

idx = s.index.to_series()
run = (idx.diff() > pd.Timedelta("30s")).cumsum()
runs = idx.groupby(run.values).agg(start="first", end="last", points="count")
print(f"\nunbroken runs: {len(runs)}")
print(runs.to_string(index=False))