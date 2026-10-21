import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from src.logging_config import log
from src.models.base import Forecaster


class Net(nn.Module):
    def __init__(self, hidden: int, horizon: int):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, horizon)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h[:, -1])

class LSTMForecaster(Forecaster):
    """Direct multi step LSTM last `lookback` points in, next `horizon` points out."""

    name = "lstm"

    def __init__(self, horizon=12, lookback=24, hidden=32, epochs=200, lr=1e-2, seed=0):
        self.horizon = horizon
        self.lookback = lookback
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.net = None

    def fit(self, series: pd.Series) -> None:
        vals = series.to_numpy(dtype="float32")
        need = self.lookback + self.horizon
        if len(vals) <= need:
            raise ValueError(f"need more than {need} points, got {len(vals)}")

        self.step = series.index.to_series().diff().median()
        self.last_ts = series.index[-1]
        self.mu = float(vals.mean())
        self.sd = float(vals.std()) or 1.0
        z = (vals - self.mu) / self.sd

        wins = sliding_window_view(z, need)
        x = torch.tensor(wins[:, : self.lookback]).unsqueeze(-1)
        y = torch.tensor(wins[:, self.lookback :])

        torch.manual_seed(self.seed)
        self.net = Net(self.hidden, self.horizon)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = nn.functional.mse_loss(self.net(x), y)
            loss.backward()
            opt.step()

        self.tail = z[-self.lookback :]
        log.info("lstm_fit_done", n_points=len(series), loss=round(loss.item(), 4))

    def predict(self, steps: int) -> pd.Series:
        if self.net is None:
            raise RuntimeError("call fit() before predict()")
        if steps > self.horizon:
            raise ValueError(f"steps={steps} exceeds horizon={self.horizon}")

        with torch.no_grad():
            x = torch.tensor(self.tail).view(1, -1, 1)
            z = self.net(x)[0, :steps].numpy()

        idx = pd.date_range(self.last_ts + self.step, periods=steps, freq=self.step)
        return pd.Series(z * self.sd + self.mu, index=idx, name="predicted_mean")
