import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    backtest_runs,
    median_ratio,
    ratio_interval,
    rolling_backtest,
    score,
    win_rate,
)
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

def test_median_ratio_ignores_one_blowup():
    res = pd.DataFrame(
        {"mae": [1.0, 1.0, 1e9], "naive_mae": [2.0, 2.0, 2.0]}
    )
    assert median_ratio(res) == pytest.approx(0.5)
    assert score(res) > 1e8

def test_seasonal_naive_is_exact_on_a_periodic_series():
    idx = pd.date_range("2026-01-01", periods=120, freq="10s")
    s = pd.Series(np.sin(2 * np.pi * np.arange(120) / 12), index=idx)

    res = rolling_backtest(s, RepeatLast, period=12)

    assert res["snaive_mae"].max() < 1e-9
    
def test_win_rate_counts_windows_below_baseline():
    res = pd.DataFrame({"mae": [1.0, 3.0, 1.0, 1.0], "snaive_mae": [2.0, 2.0, 2.0, 2.0]})
    assert win_rate(res) == 0.75


def test_interval_collapses_when_every_window_is_identical():
    res = pd.DataFrame({"mae": [1.0] * 40, "snaive_mae": [2.0] * 40})
    lo, hi = ratio_interval(res)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)