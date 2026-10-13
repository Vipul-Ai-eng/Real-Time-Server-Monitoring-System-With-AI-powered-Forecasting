import pandas as pd

from src.pre_processing import clean_series, latest_contiguous, split_runs, to_training_frame


def test_resamples_onto_fixed_grid():
    idx = pd.date_range("2026-01-01", periods=5, freq="10s")
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

    out = clean_series(s, freq="10s")

    assert len(out) == 5
    assert out.index.freq is not None


def test_fills_a_short_gap():
    idx = pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T00:00:10", "2026-01-01T00:00:30"])
    s = pd.Series([1.0, 2.0, 4.0], index=idx)

    out = clean_series(s, freq="10s", max_gap_fill=6)

    # missing 00:00:20 should land around 3.0 after interpolation
    assert len(out) == 4
    assert out.iloc[2] == 3.0


def test_leaves_most_of_a_long_gap_unfilled():
    idx = pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T00:20:00"])
    s = pd.Series([1.0, 2.0], index=idx)

    out = clean_series(s, freq="10s", max_gap_fill=6)

    # 120 missing steps only 6 get interpolated near start - the rest stay NaN and drop
    assert len(out) == 8


def test_empty_series_stays_empty():
    s = pd.Series(dtype=float)
    out = clean_series(s)
    assert len(out) == 0


def test_duplicate_timestamp_keeps_last_value():
    idx = pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T00:00:00"])
    s = pd.Series([1.0, 2.0], index=idx)

    out = clean_series(s, freq="10s")

    assert len(out) == 1
    assert out.iloc[0] == 2.0


def test_training_frame_has_expected_shape():
    idx = pd.date_range("2026-01-01", periods=3, freq="10s")
    s = pd.Series([1.0, 2.0, 3.0], index=idx)

    df = to_training_frame(s)

    assert list(df.columns) == ["value"]
    assert df.index.name == "timestamp"
    assert len(df) == 3

def test_latest_contiguous_drops_points_before_a_break():
    first = pd.date_range("2026-01-01 00:00:00", periods=3, freq="10s")
    second = pd.date_range("2026-01-01 00:10:00", periods=4, freq="10s")
    s = pd.Series(range(7), index=first.append(second), dtype=float)

    out = latest_contiguous(s, freq="10s")

    assert len(out) == 4
    assert out.index.freq is not None

def test_split_runs_breaks_on_gap_and_drops_short_ones():
    first = pd.date_range("2026-01-01 00:00", periods=5, freq="10s")
    second = pd.date_range("2026-01-01 01:00", periods=3, freq="10s")
    s = pd.Series(range(8), index=first.append(second), dtype=float)

    assert [len(r) for r in split_runs(s)] == [5, 3]
    assert len(split_runs(s, min_len=4)) == 1

