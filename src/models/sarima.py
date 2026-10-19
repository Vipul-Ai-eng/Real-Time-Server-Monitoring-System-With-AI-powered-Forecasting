import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.logging_config import log
from src.models.base import Forecaster


class SarimaForecaster(Forecaster):
    name = "sarima"

    def __init__(self, seasonal_periods: int = 12, order= (1, 1, 1)):
        self.seasonal_periods = seasonal_periods
        self.order = order
        self.seasonal_order = (1, 1, 1, seasonal_periods)
        self.fitted = None
    def fit(self, series: pd.Series) -> None:
        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True, 
        )
        self.fitted = model.fit(disp = False)
        log.info("sarima_fit_done", aic = round(self.fitted.aic, 2), n_points=len(series))
    def predict(self, steps: int) -> pd.Series:
        if self.fitted is None:
            raise RuntimeError("call fit() before predict()")
        return self.fitted.forecast(steps= steps)
