import importlib
import sys

from src.config import Settings


def test_config_module_imports_without_required_env(monkeypatch):
    monkeypatch.delenv("INFLUXDB_TOKEN", raising=False)
    monkeypatch.delenv("INFLUXDB_ORG", raising=False)

    sys.modules.pop("src.config", None)
    importlib.import_module("src.config")


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("INFLUXDB_TOKEN", "test-token")
    monkeypatch.setenv("INFLUXDB_ORG", "test-org")
    monkeypatch.setenv("INFLUXDB_BUCKET", "test-bucket")

    settings = Settings(_env_file=None)

    assert settings.influxdb_token == "test-token"
    assert settings.influxdb_org == "test-org"
    assert settings.influxdb_bucket == "test-bucket"


def test_settings_has_sensible_defaults(monkeypatch):
    monkeypatch.setenv("INFLUXDB_TOKEN", "test-token")
    monkeypatch.setenv("INFLUXDB_ORG", "test-org")

    settings = Settings(_env_file=None)

    assert settings.forecast_steps == 48
    assert settings.seasonal_periods == 12
    assert settings.influxdb_bucket == "metrics"