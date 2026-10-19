import pandas as pd

from src.backtest import backtest_runs
from src.config import get_settings
from src.ingestion import fetch_metric_series
from src.models.sarima import SarimaForecaster
from src.pre_processing import clean_series, split_runs

cfg = get_settings()
raw = fetch_metric_series("node_memory_MemAvailable_bytes", minutes=4320) # 3days = 4320 minutes
series = clean_series(raw)
if cfg.data_start:
    series = series.loc[pd.Timestamp(cfg.data_start):]

runs = split_runs(series, min_len=72)
res = backtest_runs(runs, lambda: SarimaForecaster(seasonal_periods=cfg.seasonal_periods))
res["ratio"] = res["mae"] / res["naive_mae"]

print(res.sort_values("ratio", ascending=False).head(5).to_string())
print("median ratio:", res["ratio"].median())
print("windows above 2x:", (res["ratio"] > 2).sum(), "of", len(res))