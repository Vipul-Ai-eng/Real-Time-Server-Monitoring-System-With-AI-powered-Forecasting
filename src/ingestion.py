import warnings

import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction

from src.config import get_settings
from src.logging_config import log

warnings.simplefilter("ignore", MissingPivotFunction)
def _frame_to_series(df: pd.DataFrame) -> pd.Series:
    """Convert a raw InfluxDB query_data_frame result into a clean, time indexed Series."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df = df.sort_values("_time")
    series = pd.Series(df["_value"].values, index=pd.to_datetime(df["_time"]))
    series.index.name = "timestamp"
    return series
def fetch_metric_series(measurement: str, field: str = "gauge", minutes: int = 60) -> pd.Series:
    """Fetch a single metric time series from InfluxDB over the last minutes"""
    settings = get_settings()

    flux_query = f"""
    from(bucket: "{settings.influxdb_bucket}")
      |> range(start: -{minutes}m)
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> filter(fn: (r) => r._field == "{field}")
    """

    with InfluxDBClient(
        url=settings.influxdb_url,
        token=settings.influxdb_token,
        org=settings.influxdb_org,
    ) as client:
        query_api = client.query_api()
        df = query_api.query_data_frame(flux_query)

    if isinstance(df, list):
        df = pd.concat(df) if df else pd.DataFrame()

    series = _frame_to_series(df)
    log.info("metric_fetched", measurement=measurement, points=len(series))
    return series


if __name__ == "__main__":
    from src.logging_config import configure_logging

    configure_logging()
    result = fetch_metric_series("node_memory_MemAvailable_bytes", minutes=30)
    print(result.tail())