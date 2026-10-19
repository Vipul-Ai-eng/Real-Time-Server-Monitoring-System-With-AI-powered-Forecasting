import pandas as pd

from src.backtest import HORIZON, MIN_TRAIN, backtest_runs, median_ratio, score
from src.config import get_settings
from src.ingestion import fetch_metric_series
from src.logging_config import configure_logging
from src.models.prophet_model import ProphetForecaster
from src.models.sarima import SarimaForecaster
from src.pre_processing import clean_series, split_runs

configure_logging()
cfg = get_settings()

raw = fetch_metric_series("node_memory_MemAvailable_bytes", minutes=4320) # 3days = 4320 minutes
series = clean_series(raw)
if cfg.data_start:
    series = series.loc[pd.Timestamp(cfg.data_start):]

runs = split_runs(series, min_len=MIN_TRAIN + HORIZON)
print(f"\nusable runs: {len(runs)}, points: {sum(len(r) for r in runs)}")

models = {
    "sarima": lambda: SarimaForecaster(seasonal_periods=cfg.seasonal_periods),
    "prophet": ProphetForecaster,
    "prophet+cycle": lambda: ProphetForecaster(period_steps=cfg.seasonal_periods),
}

for name, make in models.items():
    res = backtest_runs(runs, make, period=cfg.seasonal_periods)
    print(
        f"{name}: {len(res)} windows, "
        f"{score(res):.2f}x naive, {score(res, 'snaive_mae'):.2f}x seasonal-naive, "
        f"{median_ratio(res, 'snaive_mae'):.2f}x median vs seasonal"
    )

print(f"seasonal-naive itself: {res['snaive_mae'].sum() / res['naive_mae'].sum():.2f}x naive")