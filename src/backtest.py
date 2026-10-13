import numpy as np
import pandas as pd


def rolling_backtest(series, make_model, horizon=12, min_train=60, step=12):
    """Score a model on successive windows, each trained only on earlier data"""
    rows = []
    end = min_train
    while end + horizon <= len(series):
        train = series.iloc[:end]
        test = series.iloc[end : end + horizon]

        model = make_model()
        model.fit(train)
        pred = model.predict(horizon)

        rows.append(
            {
                "end": train.index[-1],
                "mae": np.abs(test.values - pred.values).mean(),
                "naive_mae": np.abs(test.values - train.iloc[-1]).mean(),
            }
        )
        end += step
    return pd.DataFrame(rows)


def score(results):
    """total model error over total naive error. Below 1.0 beats repeat last value"""
    if results.empty:
        return float("nan")
    return results["mae"].sum() / results["naive_mae"].sum()

def backtest_runs(runs, make_model, horizon=12, min_train=60, step=12):
    """Backtest each run separately and stack the results"""
    parts = [rolling_backtest(r, make_model, horizon, min_train, step) for r in runs]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["end", "mae", "naive_mae"])
    return pd.concat(parts, ignore_index=True)