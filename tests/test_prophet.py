import numpy as np
import pandas as pd
import pytest

from src.models.prophet_model import ProphetForecaster


def _fake_series(n=100):
    idx = pd.date_range("2026-01-01", periods=n, freq="10s", tz="UTC")
    vals = 50 + 5 * np.sin(np.arange(n) * 2 * np.pi / 12)
    return pd.Series(vals, index=idx)


def test_predict_returns_right_number_of_steps():
    model = ProphetForecaster()
    model.fit(_fake_series())
    assert len(model.predict(steps=10)) == 10


def test_forecast_continues_after_last_point():
    s = _fake_series()
    model = ProphetForecaster()
    model.fit(s)
    out = model.predict(steps=5)

    assert out.index[0] == s.index[-1] + pd.Timedelta("10s")


def test_forecast_keeps_timezone():
    model = ProphetForecaster()
    model.fit(_fake_series())
    out = model.predict(steps=3)

    assert str(out.index.tz) == "UTC"


def test_predict_before_fit_raises():
    model = ProphetForecaster()
    with pytest.raises(RuntimeError):
        model.predict(steps=5)

