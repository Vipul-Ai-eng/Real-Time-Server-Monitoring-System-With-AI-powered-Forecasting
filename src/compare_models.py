import numpy as np

from src.config import get_settings
from src.ingestion import fetch_metric_series
from src.logging_config import configure_logging, log
from src.models.prophet_model import ProphetForecaster
from src.models.sarima import SarimaForecaster
from src.pre_processing import clean_series, latest_contiguous

HOLDOUT = 12

configure_logging()
cfg = get_settings()

raw = fetch_metric_series("node_memory_MemAvailable_bytes", minutes=60)
series = latest_contiguous(clean_series(raw))
train, test = series.iloc[:-HOLDOUT], series.iloc[-HOLDOUT:]
log.info("split_ready", train=len(train), test=len(test))

naive_mae = np.abs(test.values - train.iloc[-1]).mean()
print(f"\nnaive (repeat last value) MAE: {naive_mae:,.0f}")

for model in [SarimaForecaster(seasonal_periods=cfg.seasonal_periods), ProphetForecaster()]:
    model.fit(train)
    pred = model.predict(HOLDOUT)
    mae = np.abs(test.values - pred.values).mean()
    print(f"{model.name} MAE: {mae:,.0f}  ({mae / naive_mae:.2f}x naive)")