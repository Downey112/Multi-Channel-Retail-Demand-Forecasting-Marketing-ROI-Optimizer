import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mcr import analysis as A  # noqa: E402
from mcr.calendar import festive_weeks, monthly_cpi, weekly_cpi, weeks  # noqa: E402
from mcr.simulate import TRUE, simulate  # noqa: E402  (tests may use the truth; the analysis may not)
from mcr.transforms import adbudg_inflection, adbudg_shape, adstock  # noqa: E402

CPI_FILE = ROOT / "data" / "raw" / "cpi_2d.parquet"
needs_cpi = pytest.mark.skipif(not CPI_FILE.exists(), reason="run scripts/01_simulate.py first")


def test_adstock_geometric_carryover():
    assert adstock(np.array([1.0, 0, 0, 2.0]), 0.5).tolist() == [1.0, 0.5, 0.25, 2.125]


def test_adbudg_inflection_is_where_curvature_changes_sign():
    c, h = 1.8, 45.0
    x0 = adbudg_inflection(c, h)
    d2 = lambda x, e=1e-3: (adbudg_shape(x + e, c, h) - 2 * adbudg_shape(x, c, h) + adbudg_shape(x - e, c, h)) / e ** 2  # noqa: E731
    assert d2(x0 * 0.9) > 0 > d2(x0 * 1.1)
    assert adbudg_shape(h, c, h) == pytest.approx(0.5)


def test_festive_weeks_cover_the_week_and_the_week_before():
    f = festive_weeks(weeks("2024-01-01", 10))
    assert f[pd.Timestamp("2024-02-05")] == "Chinese New Year"      # CNY 2024-02-10 is a Saturday
    assert f[pd.Timestamp("2024-01-29")] == "Chinese New Year"
    assert f[pd.Timestamp("2024-02-12")] == ""


def test_weekly_cpi_refuses_to_invent_missing_months():
    cpi = pd.Series([130.0], index=pd.to_datetime(["2024-01-01"]))
    assert weekly_cpi(weeks("2024-01-01", 4), cpi).eq(130.0).all()
    with pytest.raises(ValueError):
        weekly_cpi(weeks("2024-01-22", 4), cpi)                  # runs into February


@needs_cpi
def test_simulation_is_reproducible_and_blocks_are_balanced():
    cpi = monthly_cpi(CPI_FILE)
    a, b, c = simulate(cpi), simulate(cpi), simulate(cpi, seed=1)
    assert a["weekly_sales"].equals(b["weekly_sales"])
    assert not a["weekly_sales"]["revenue_rm"].equals(c["weekly_sales"]["revenue_rm"])
    bal = a["promo_assignments"].groupby(["store_id", "block_id"])["promo_type"].agg(["size", "nunique"])
    assert (bal["size"] == 4).all() and (bal["nunique"] == 4).all()


def _noiseless(**extra):
    t = {**TRUE, "chain_shock_sd_rm": 0.0, "store_noise_sd_rm": 0.0, **extra}
    return simulate(monthly_cpi(CPI_FILE), truth=t)


@needs_cpi
def test_promo_simple_effects_recover_the_truth_without_noise():
    exp = A.promo_experiment(A.store_week_frame(_noiseless()))
    for d in exp["simple_effects"]:
        true = TRUE["promo_rm"][d["promo"]] + TRUE["interaction_rm"].get((d["region"], d["promo"]), 0.0)
        assert d["lift_rm"] == pytest.approx(true, abs=300)       # left-over: store-specific share of media


@needs_cpi
def test_media_response_recovers_the_truth_without_noise():
    # also without region x promo interactions, which the chain-level model (promo counts only) omits
    fit = A.fit_response(A.chain_week_frame(_noiseless(interaction_rm={})))
    th = fit["theta"]
    assert th["adstock_lambda"] == pytest.approx(TRUE["adstock_lambda"], abs=1e-3)
    assert th["c"] == pytest.approx(TRUE["adbudg_c"], rel=1e-3)
    assert th["half_k"] == pytest.approx(TRUE["adbudg_half_k"], rel=1e-3)
    assert [th["w_search_sem"], th["w_in_app_push"], th["w_out_of_home"]] == pytest.approx([1.3, 0.6, 0.5], rel=1e-3)
    assert fit["beta"]["ceiling_rm"] == pytest.approx(TRUE["adbudg_max_rm"], rel=1e-3)
    assert fit["beta"]["festive"] == pytest.approx(12 * TRUE["festive_rm"], rel=1e-3)


def test_profit_optimal_budget_sets_marginal_profit_to_one():
    theta = {"adstock_lambda": 0.45, "c": 1.8, "half_k": 45.0, "w_search_sem": 1.3, "w_in_app_push": 0.6,
             "w_out_of_home": 0.5}
    mix = {"Social_Video": 0.4, "Search_SEM": 0.3, "In_App_Push": 0.1, "Out_Of_Home": 0.2}
    r = A.roi_summary(theta, 240_000.0, mix, margin=0.35)
    s, h = r["optimal_spend_k"], 1e-3
    marginal = (A.long_run_lift(s + h, theta, 240_000.0, mix) - A.long_run_lift(s - h, theta, 240_000.0, mix)) / (2 * h)
    assert 0.35 * marginal / 1000 == pytest.approx(1.0, abs=1e-4)
    assert r["optimal_spend_k"] > r["inflection_spend_k"]
