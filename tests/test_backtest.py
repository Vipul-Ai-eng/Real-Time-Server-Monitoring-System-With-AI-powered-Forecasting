import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_runs, rolling_backtest, score
from src.models.base import Forecaster


class RepeatLast(Forecaster):
    name = "repeat_last"

    def fit(self, series):
        self.last = series.iloc[-1]
        self.end = series.index[-1]

    def predict(self, steps):
        idx = pd.date_range(self.end + pd.Timedelta("10s"), periods=steps, freq="10s")
        return pd.Series([self.last] * steps, index=idx)


def _series(n):
    idx = pd.date_range("2026-01-01", periods=n, freq="10s")
    return pd.Series(np.sin(np.arange(n) / 3), index=idx)


def test_window_count():
    res = rolling_backtest(_series(100), RepeatLast, horizon=12, min_train=60, step=12)
    assert len(res) == 3


def test_too_short_series_gives_empty_result():
    res = rolling_backtest(_series(50), RepeatLast, horizon=12, min_train=60)
    assert res.empty
    assert np.isnan(score(res))


def test_repeat_last_scores_exactly_naive():
    res = rolling_backtest(_series(100), RepeatLast)
    assert score(res) == pytest.approx(1.0)

def test_backtest_runs_never_crosses_a_gap():
    long_run = _series(100)
    short_run = _series(50)
    short_run.index = short_run.index + pd.Timedelta("1D")

    res = backtest_runs([long_run, short_run], RepeatLast)

    assert len(res) == 3