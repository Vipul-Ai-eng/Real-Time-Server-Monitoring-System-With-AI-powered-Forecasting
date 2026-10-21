import numpy as np
import pandas as pd

HORIZON = 12
MIN_TRAIN = 60
STEP = 12


def rolling_backtest(
    series, make_model, horizon=HORIZON, min_train=MIN_TRAIN, step=STEP, period=None
):
    """Score a model on successive windows, each trained only on earlier data.

    With `period` set, also scores a seasonal-naive baseline that repeats the
    last full cycle of the training data.
    """
    rows = []
    end = min_train
    while end + horizon <= len(series):
        train = series.iloc[:end]
        test = series.iloc[end : end + horizon]

        model = make_model()
        model.fit(train)
        pred = model.predict(horizon)

        snaive_mae = np.nan
        if period:
            cycle = np.resize(train.values[-period:], horizon)
            snaive_mae = np.abs(test.values - cycle).mean()

        rows.append(
            {
                "end": train.index[-1],
                "mae": np.abs(test.values - pred.values).mean(),
                "naive_mae": np.abs(test.values - train.iloc[-1]).mean(),
                "snaive_mae": snaive_mae,
            }
        )
        end += step
    return pd.DataFrame(rows)


def score(results, base="naive_mae"):
    """Total model error over total baseline error. Below 1.0 beats the baseline."""
    if results.empty:
        return float("nan")
    return results["mae"].sum() / results[base].sum()


def median_ratio(results, base="naive_mae"):
    """Median per-window error over baseline error, robust to a few bad fits."""
    if results.empty:
        return float("nan")
    return (results["mae"] / results[base]).median()


def backtest_runs(
    runs, make_model, horizon=HORIZON, min_train=MIN_TRAIN, step=STEP, period=None
):
    """Backtest each run separately and stack the results."""
    parts = [
        rolling_backtest(r, make_model, horizon, min_train, step, period) for r in runs
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["end", "mae", "naive_mae", "snaive_mae"])
    return pd.concat(parts, ignore_index=True)

def win_rate(results, base="snaive_mae"):
    """Share of windows where the model error is below the baseline"""
    if results.empty:
        return float("nan")
    return float((results["mae"] < results[base]).mean())

def ratio_interval(results, base="snaive_mae", n=2000, seed=0):
    """Bootstrap 90% interval for total model error over total baseline error"""
    if results.empty:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = results["mae"].to_numpy()
    b = results[base].to_numpy()
    pick = rng.integers(0, len(m), size=(n, len(m)))
    lo, hi = np.percentile(m[pick].sum(axis=1) / b[pick].sum(axis=1), [5, 95])
    return float(lo), float(hi)

def since(results, start):
    """Keep windows whose training data ends at or after start"""
    if start is None or results.empty:
        return results
    return results[results["end"] >= pd.Timestamp(start)].reset_index(drop=True)