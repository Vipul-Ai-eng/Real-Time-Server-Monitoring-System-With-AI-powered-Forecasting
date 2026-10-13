import pandas as pd

from src.logging_config import log


def clean_series(raw: pd.Series, freq: str = "10s", max_gap_fill: int = 6) -> pd.Series:
    """resample to a fixed frequency and fill small gaps, dropping anything unfillable
    """
    if raw.empty:
        return raw

    no_dupes = raw[~raw.index.duplicated(keep="last")]
    ordered = no_dupes.sort_index()

    grid = ordered.resample(freq).mean()
    filled = grid.interpolate(method="linear", limit=max_gap_fill)
    clean = filled.dropna()

    log.info(
        "series_cleaned",
        raw_points=len(raw),
        resampled_points=len(grid),
        final_points=len(clean),
    )
    return clean

def latest_contiguous(series: pd.Series, freq: str = "10s") -> pd.Series:
    """Return the most recent unbroken run of points, indexed at a fixed freq."""
    if series.empty:
        return series

    gap = series.index.to_series().diff() > pd.Timedelta(freq)
    start = gap[gap].index[-1] if gap.any() else series.index[0]
    return series.loc[start:].asfreq(freq)

def to_training_frame(series: pd.Series) -> pd.DataFrame:
    """Shape cleaned series into DataFrame our models expect"""
    df = series.to_frame(name="value")
    df.index.name = "timestamp"
    return df

def split_runs(series: pd.Series, freq: str = "10s", min_len: int = 1) -> list[pd.Series]:
    """Split a series into unbroken runs, starting a new run after any gap."""
    if series.empty:
        return []

    run_id = (series.index.to_series().diff() > pd.Timedelta(freq)).cumsum()
    runs = [g.asfreq(freq) for _, g in series.groupby(run_id.values)]
    return [r for r in runs if len(r) >= min_len]

if __name__ == "__main__":
    from src.ingestion import fetch_metric_series
    from src.logging_config import configure_logging

    configure_logging()
    raw = fetch_metric_series("node_memory_MemAvailable_bytes", minutes=30)
    clean = clean_series(raw)
    print(clean.tail())