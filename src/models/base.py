from abc import ABC, abstractmethod

import pandas as pd


class Forecaster(ABC):
    """Common shape every forecasting backend follows so they are swappable"""
    name: str

    @abstractmethod
    def fit(self, series: pd.Series) -> None:
        """Fit on historical series"""

    @abstractmethod
    def predict(self, steps: int) -> pd.Series:
        """return steps"""