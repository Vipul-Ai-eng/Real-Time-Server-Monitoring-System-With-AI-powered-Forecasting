import numpy as np
import pandas as pd
import pytest

from src.models.lstm import LSTMForecaster


def _fake_series(n=200):
    idx = pd.date_range("2026-01-01", periods=n, freq="10s", tz="UTC")
    vals = 50 + 5 * np.sin(np.arange(n) * 2 * np.pi / 12)
    return pd.Series(vals, index=idx)


def test_forecast_shape_index_and_timezone():
    s = _fake_series()
    model = LSTMForecaster()
    model.fit(s)
    out = model.predict(12)

    assert len(out) == 12
    assert out.index[0] == s.index[-1] + pd.Timedelta("10s")
    assert str(out.index.tz) == "UTC"


def test_same_seed_gives_same_forecast():
    a, b = LSTMForecaster(), LSTMForecaster()
    a.fit(_fake_series())
    b.fit(_fake_series())

    assert np.allclose(a.predict(12).values, b.predict(12).values)


def test_steps_beyond_horizon_raises():
    model = LSTMForecaster(horizon=12)
    model.fit(_fake_series())
    with pytest.raises(ValueError):
        model.predict(13)


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        LSTMForecaster().fit(_fake_series(30))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LSTMForecaster().predict(5)


def test_learns_a_known_wave():
    s = _fake_series(212)
    train, test = s.iloc[:200], s.iloc[200:]

    model = LSTMForecaster()
    model.fit(train)
    pred = model.predict(12)

    assert np.abs(pred.values - test.values).mean() < 1.5