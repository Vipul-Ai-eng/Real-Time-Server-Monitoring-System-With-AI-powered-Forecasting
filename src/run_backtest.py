import pandas as pd

from src.backtest import backtest_runs, score
from src.config import get_settings
from src.ingestion import fetch_metric_series
from src.logging_config import configure_logging
from src.models.prophet_model import ProphetForecaster
from src.models.sarima import SarimaForecaster
from src.pre_processing import clean_series, split_runs

configure_logging()
cfg = get_settings()

raw = fetch_metric_series("node_memory_MemAvailable_bytes", minutes=4320)
series = clean_series(raw)
if cfg.data_start:
    series = series.loc[pd.Timestamp(cfg.data_start):]

runs = split_runs(series, min_len=72)
print(f"\nusable runs: {len(runs)}, points: {sum(len(r) for r in runs)}")

models = {
    "sarima": lambda: SarimaForecaster(seasonal_periods=cfg.seasonal_periods),
    "prophet": ProphetForecaster,
}
for name, make in models.items():
    res = backtest_runs(runs, make)
    print(f"{name}: {len(res)} windows, {score(res):.2f}x naive")