import logging

import pandas as pd
from prophet import Prophet

from src.logging_config import log
from src.models.base import Forecaster

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


class ProphetForecaster(Forecaster):
    name = "prophet"

    def __init__(self, changepoint_scale: float = 0.05):
        self.changepoint_scale = changepoint_scale
        self.model = None
        self.step = None
        self.last_ts = None
        self.tz = None

    def fit(self, series: pd.Series) -> None:
        self.tz = series.index.tz
        self.step = series.index.to_series().diff().median()

        ds = series.index.tz_localize(None)
        self.last_ts = ds[-1]

        df = pd.DataFrame({"ds": ds, "y": series.values})
        self.model = Prophet(changepoint_prior_scale=self.changepoint_scale)
        self.model.fit(df)
        log.info("prophet_fit_done", n_points=len(series))

    def predict(self, steps: int) -> pd.Series:
        if self.model is None:
            raise RuntimeError("call fit() before predict()")

        idx = pd.date_range(self.last_ts + self.step, periods=steps, freq=self.step)
        out = self.model.predict(pd.DataFrame({"ds": idx}))

        if self.tz is not None:
            idx = idx.tz_localize(self.tz)
        return pd.Series(out["yhat"].values, index=idx, name="predicted_mean")