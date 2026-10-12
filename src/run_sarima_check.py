from src.config import get_settings
from src.ingestion import fetch_metric_series
from src.logging_config import configure_logging, log
from src.models.sarima import SarimaForecaster
from src.pre_processing import clean_series, latest_contiguous

configure_logging()
cfg = get_settings()

raw = fetch_metric_series("node_memory_MemAvailable_bytes", minutes=60)
clean = latest_contiguous(clean_series(raw))
log.info("data_ready", points=len(clean))

model = SarimaForecaster(seasonal_periods=cfg.seasonal_periods)
model.fit(clean)
forecast = model.predict(steps=12)

print("\nlast actual values-")
print(clean.tail(3))
print("\nforecast (next 12 steps)-")
print(forecast)