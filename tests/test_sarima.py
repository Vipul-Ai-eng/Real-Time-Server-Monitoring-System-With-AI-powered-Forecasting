import numpy as np
import pandas as pd

from src.models.sarima import SarimaForecaster


def _fake_series(n=100):
    idx = pd.date_range("2026-01-01", periods=n, freq="10s")
    values = 50 + 5 * np.sin(np.arange(n) * 2 * np.pi / 12)
    return pd.Series(values, index=idx)


def test_fit_sets_fitted_model():
    model = SarimaForecaster(seasonal_periods=12)
    model.fit(_fake_series())
    assert model.fitted is not None


def test_predict_returns_right_number_of_steps():
    model = SarimaForecaster(seasonal_periods=12)
    model.fit(_fake_series())
    forecast = model.predict(steps=10)
    assert len(forecast) == 10


def test_predict_before_fit_raises():
    import pytest

    model = SarimaForecaster(seasonal_periods=12)
    with pytest.raises(RuntimeError):
        model.predict(steps=5)