from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str
    influxdb_org: str
    influxdb_bucket: str = "metrics"

    mlflow_tracking_uri: str = "http://localhost:5000"

    forecast_steps: int = 48        # 4 hours at 5 minute intervals
    seasonal_periods: int = 12      # 1 hour seasonality


