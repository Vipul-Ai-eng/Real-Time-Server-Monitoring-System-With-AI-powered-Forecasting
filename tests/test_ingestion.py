import pandas as pd

from src.ingestion import _frame_to_series


def test_frame_to_series_converts_correctly():
    df = pd.DataFrame(
        {
            "_time": ["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"],
            "_value": [42.0, 43.5],
        }
    )
    series = _frame_to_series(df)

    assert len(series) == 2
    assert series.iloc[0] == 42.0
    assert series.iloc[1] == 43.5


def test_frame_to_series_sorts_by_time():
    df = pd.DataFrame(
        {
            "_time": ["2026-01-01T00:05:00Z", "2026-01-01T00:00:00Z"],
            "_value": [43.5, 42.0],
        }
    )
    series = _frame_to_series(df)

    assert series.iloc[0] == 42.0
    assert series.iloc[1] == 43.5


def test_frame_to_series_handles_empty_input():
    df = pd.DataFrame(columns=["_time", "_value"])
    series = _frame_to_series(df)

    assert len(series) == 0


def test_frame_to_series_handles_none():
    series = _frame_to_series(None)
    assert len(series) == 0