"""Invariant checks against the loaded database (skipped if PostgreSQL is not reachable)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mcr.analysis import chain_week_frame  # noqa: E402
from mcr.calendar import monthly_cpi  # noqa: E402
from mcr.db import connect, engine  # noqa: E402

try:
    _conn = connect()
except Exception:  # pragma: no cover
    pytest.skip("PostgreSQL not reachable; docker compose up -d", allow_module_level=True)


def one(sql):
    return _conn.execute(sql).fetchone()


def test_row_counts_and_keys():
    assert one("SELECT count(*) FROM stores")[0] == 12
    assert one("SELECT count(*) FROM weeks")[0] == 156
    assert one("SELECT count(*) FROM marketing_spend")[0] == 156 * 4
    assert one("SELECT count(*) FROM weekly_sales")[0] == 12 * 156
    assert one("SELECT count(*) FROM weekly_sales ws LEFT JOIN promo_assignments pa USING (store_id, week_start) "
               "WHERE pa.promo_type IS NULL")[0] == 0


def test_randomised_complete_blocks():
    bad = one("SELECT count(*) FROM vw_rcbd_balance WHERE weeks <> 4 OR distinct_promos <> 4")[0]
    assert bad == 0 and one("SELECT count(*) FROM vw_rcbd_balance")[0] == 12 * 39


def test_chain_view_matches_an_independent_pandas_build():
    sim = ROOT / "data" / "sim"
    t = {n: pd.read_parquet(sim / f"{n}.parquet") for n in
         ("stores", "weeks", "marketing_spend", "promo_assignments", "weekly_sales")}
    py = chain_week_frame(t)
    db = pd.read_sql("SELECT * FROM vw_chain_week ORDER BY week_start", engine())
    assert len(py) == len(db) == 156
    for col in ["revenue_rm", "cpi_index", "total_spend_k", "spend_search_sem_k", "n_flash_sale", "festive"]:
        assert np.allclose(py[col].astype(float), db[col].astype(float), atol=0.01), col


def test_cpi_is_the_real_opendosm_series():
    cpi = monthly_cpi(ROOT / "data" / "raw" / "cpi_2d.parquet")
    db = pd.read_sql("SELECT week_start, cpi_index FROM weeks", engine())
    month = pd.to_datetime(db["week_start"]).dt.to_period("M").dt.to_timestamp()
    assert np.allclose(db["cpi_index"].astype(float), cpi.reindex(month).to_numpy())


def test_festive_weeks_and_mondays():
    assert one("SELECT count(*) FROM weeks WHERE festival <> ''")[0] == 24      # 4 festivals x 3 years x 2 weeks
    assert one("SELECT count(DISTINCT festival) FROM weeks WHERE festival <> ''")[0] == 4


def test_latest_forecast_run_is_complete():
    run = one("SELECT max(run_id) FROM model_runs")[0]
    if run is None:
        pytest.skip("run scripts/05_forecast.py first")
    n, first, last, gaps = one(f"""
        SELECT count(*), min(week_start), max(week_start),
               count(*) FILTER (WHERE week_start - lag_w <> 7)
        FROM (SELECT week_start, LAG(week_start) OVER (PARTITION BY scenario ORDER BY week_start) AS lag_w
              FROM forecasts WHERE run_id = {run}) f""")
    trained = one(f"SELECT trained_through FROM model_runs WHERE run_id = {run}")[0]
    assert n == 26 and gaps == 0 and (first - trained).days == 7
