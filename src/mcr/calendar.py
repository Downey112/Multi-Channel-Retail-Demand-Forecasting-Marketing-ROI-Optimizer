"""Real exogenous inputs: Malaysian festive calendar and OpenDOSM monthly CPI, on a weekly grid."""
from pathlib import Path

import numpy as np
import pandas as pd

CPI_URL = "https://storage.dosm.gov.my/cpi/cpi_2d.parquet"

# Public-holiday dates (first day). Retail demand peaks in the run-up, so the festive flag covers
# the week containing the date and the week before it.
FESTIVALS = {
    "Chinese New Year": ["2023-01-22", "2024-02-10", "2025-01-29", "2026-02-17"],
    "Hari Raya Aidilfitri": ["2023-04-22", "2024-04-10", "2025-03-31", "2026-03-21"],
    "Deepavali": ["2023-11-12", "2024-10-31", "2025-10-20"],
    "Christmas": ["2023-12-25", "2024-12-25", "2025-12-25"],
}


def weeks(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="W-MON")


def festive_weeks(week_starts: pd.DatetimeIndex) -> pd.Series:
    """Name of the festival a week belongs to (week of the date or the week before), else ''."""
    out = pd.Series("", index=week_starts, dtype=object)
    for name, dates in FESTIVALS.items():
        for d in pd.to_datetime(dates):
            wk = d - pd.Timedelta(days=d.weekday())          # Monday of the festival week
            for w in (wk - pd.Timedelta(weeks=1), wk):
                if w in out.index:
                    out[w] = name
    return out


def monthly_cpi(path: Path) -> pd.Series:
    """Headline CPI (2010 = 100) by month from the cached OpenDOSM file."""
    d = pd.read_parquet(path)
    d = d[d["division"] == "overall"]
    return d.set_index(pd.to_datetime(d["date"]))["index"].astype(float).sort_index()


def weekly_cpi(week_starts: pd.DatetimeIndex, cpi: pd.Series) -> pd.Series:
    """CPI of the calendar month each week starts in (no interpolation, no look-ahead within a month)."""
    month = week_starts.to_period("M").to_timestamp()
    vals = cpi.reindex(month).to_numpy()
    if np.isnan(vals).any():
        missing = sorted(set(month[np.isnan(vals)].strftime("%Y-%m")))
        raise ValueError(f"CPI missing for {missing}; refresh {CPI_URL}")
    return pd.Series(vals, index=week_starts, name="cpi_index")
