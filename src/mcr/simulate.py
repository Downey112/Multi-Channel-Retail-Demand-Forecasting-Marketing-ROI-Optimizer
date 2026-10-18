"""Simulated retail chain: 12 stores x 156 weeks, a randomised promo experiment and 4 marketing channels.

This is the ONLY module that knows the true parameters (TRUE). The analysis code (mcr.analysis and
scripts 03-05) must never import it; scripts/06_recovery.py compares estimates with TRUE afterwards,
and tests/test_no_leak.py enforces the separation.

Real inputs: the Malaysian festive calendar and OpenDOSM monthly CPI. Everything else is simulated,
with random noise from a seeded generator.
"""
import numpy as np
import pandas as pd

from .calendar import festive_weeks, weekly_cpi, weeks
from .transforms import adbudg_shape, adstock

PROMOS = ["Flash_Sale", "Bundle_Discount", "Loyalty_Cashback", "No_Promo"]
CHANNELS = ["Social_Video", "Search_SEM", "In_App_Push", "Out_Of_Home"]

STORES = pd.DataFrame(
    [
        ("STR_C01", "Central", "Flagship_Mall", 28000), ("STR_C02", "Central", "Hypermarket", 22000),
        ("STR_C03", "Central", "Suburban_Hub", 14500), ("STR_N01", "Northern", "Flagship_Mall", 25000),
        ("STR_N02", "Northern", "Hypermarket", 20000), ("STR_N03", "Northern", "Suburban_Hub", 13500),
        ("STR_S01", "Southern", "Flagship_Mall", 24000), ("STR_S02", "Southern", "Hypermarket", 19500),
        ("STR_S03", "Southern", "Suburban_Hub", 13000), ("STR_E01", "East_Coast", "Flagship_Mall", 21000),
        ("STR_E02", "East_Coast", "Hypermarket", 17500), ("STR_E03", "East_Coast", "Suburban_Hub", 12000),
    ],
    columns=["store_id", "region", "store_format", "floor_area_sqft"],
)

TRUE = {
    "seed": 2024,
    "start": "2023-01-02",
    "n_weeks": 156,
    # store level (RM per store-week)
    "base_rm": 40000.0,
    "per_sqft_rm": 2.0,
    "region_rm": {"Central": 18000.0, "Northern": 6000.0, "Southern": 9000.0, "East_Coast": 0.0},
    "store_sd_rm": 4000.0,                       # random store effect
    "promo_rm": {"Flash_Sale": 9000.0, "Bundle_Discount": 6000.0, "Loyalty_Cashback": 3500.0, "No_Promo": 0.0},
    "interaction_rm": {("Central", "Flash_Sale"): 3000.0, ("East_Coast", "Loyalty_Cashback"): 2000.0},
    "festive_rm": 22000.0,
    "season_sin_rm": 5000.0,
    "season_cos_rm": 2500.0,
    "cpi_rm_per_point": -1200.0,                 # relative to CPI at the first week
    # marketing (chain level, RM '000 per week)
    "channel_budget_k": {"Social_Video": 12.0, "Search_SEM": 9.0, "In_App_Push": 4.0, "Out_Of_Home": 7.0},
    "channel_weight": {"Social_Video": 1.0, "Search_SEM": 1.3, "In_App_Push": 0.6, "Out_Of_Home": 0.5},
    "spend_sd_log": 0.35,                        # independent lognormal noise per channel-week
    "festive_spend_boost": 1.6,
    "dark_weeks": 8,                             # randomly chosen weeks with no paid media
    "adstock_lambda": 0.45,
    "adbudg_max_rm": 240000.0,                   # chain-level ceiling of the media lift
    "adbudg_c": 1.8,
    "adbudg_half_k": 45.0,                       # adstocked social-equivalent spend at half the ceiling
    # noise
    "chain_ar1_phi": 0.5,
    "chain_shock_sd_rm": 1500.0,                 # per store, common to all stores in a week
    "store_noise_sd_rm": 3500.0,
    "avg_price_rm": 42.0,
}


def simulate(cpi_monthly: pd.Series, seed: int | None = None, truth: dict = TRUE) -> dict[str, pd.DataFrame]:
    """Return the tables the database holds: stores, weeks, marketing_spend, promo_assignments, weekly_sales."""
    rng = np.random.default_rng(truth["seed"] if seed is None else seed)
    wk = weeks(truth["start"], truth["n_weeks"])
    n = len(wk)
    fest = festive_weeks(wk)
    is_fest = (fest != "").to_numpy().astype(float)
    cpi = weekly_cpi(wk, cpi_monthly)
    t = np.arange(n)
    season = truth["season_sin_rm"] * np.sin(2 * np.pi * t / 52) + truth["season_cos_rm"] * np.cos(2 * np.pi * t / 52)

    # marketing spend: planned budget x festive boost x lognormal noise, some dark weeks
    dark = np.zeros(n, bool)
    dark[rng.choice(n, truth["dark_weeks"], replace=False)] = True
    spend = {}
    for ch in CHANNELS:
        s = truth["channel_budget_k"][ch] * np.where(is_fest > 0, truth["festive_spend_boost"], 1.0)
        s = s * rng.lognormal(0.0, truth["spend_sd_log"], n)
        spend[ch] = np.where(dark, 0.0, np.round(s, 3))
    effective = sum(truth["channel_weight"][ch] * spend[ch] for ch in CHANNELS)
    media_lift_chain = truth["adbudg_max_rm"] * adbudg_shape(
        adstock(effective, truth["adstock_lambda"]), truth["adbudg_c"], truth["adbudg_half_k"])

    # chain-level AR(1) shock shared by all stores in a week
    shock = np.zeros(n)
    eps = rng.normal(0.0, truth["chain_shock_sd_rm"], n)
    for i in range(n):
        shock[i] = (truth["chain_ar1_phi"] * shock[i - 1] if i else 0.0) + eps[i]

    stores = STORES.copy()
    store_fx = rng.normal(0.0, truth["store_sd_rm"], len(stores))
    base = truth["base_rm"] + truth["per_sqft_rm"] * stores["floor_area_sqft"] \
        + stores["region"].map(truth["region_rm"]) + store_fx
    share = (base / base.sum()).to_numpy()        # media lift is split in proportion to store size

    # randomised complete blocks: every store gets each promo once in each 4-week block
    n_blocks = n // 4
    promo_rows, sales_rows = [], []
    for s_idx, st in stores.iterrows():
        order = np.concatenate([rng.permutation(4) for _ in range(n_blocks)])
        for i in range(n):
            promo = PROMOS[order[i]]
            lift = truth["promo_rm"][promo] + truth["interaction_rm"].get((st["region"], promo), 0.0)
            rev = (base[s_idx] + lift + truth["festive_rm"] * is_fest[i] + season[i]
                   + truth["cpi_rm_per_point"] * (cpi.iloc[i] - cpi.iloc[0])
                   + share[s_idx] * media_lift_chain[i] + shock[i]
                   + rng.normal(0.0, truth["store_noise_sd_rm"]))
            price = truth["avg_price_rm"] * cpi.iloc[i] / cpi.iloc[0] * rng.normal(1.0, 0.02)
            promo_rows.append((st["store_id"], wk[i].date(), promo, i // 4 + 1))
            sales_rows.append((st["store_id"], wk[i].date(), round(float(rev), 2), int(round(rev / price))))

    week_df = pd.DataFrame({"week_start": wk.date, "festival": fest.to_numpy(), "cpi_index": cpi.to_numpy()})
    spend_df = pd.DataFrame(
        [(wk[i].date(), ch, spend[ch][i]) for i in range(n) for ch in CHANNELS],
        columns=["week_start", "channel", "spend_k"],
    )
    return {
        "stores": stores,
        "weeks": week_df,
        "marketing_spend": spend_df,
        "promo_assignments": pd.DataFrame(promo_rows, columns=["store_id", "week_start", "promo_type", "block_id"]),
        "weekly_sales": pd.DataFrame(sales_rows, columns=["store_id", "week_start", "revenue_rm", "units_sold"]),
    }


def truth_for_report(truth: dict = TRUE) -> dict:
    """TRUE in the units the analysis reports (JSON-serialisable)."""
    t = dict(truth)
    t["interaction_rm"] = {f"{r}:{p}": v for (r, p), v in truth["interaction_rm"].items()}
    base_w = truth["channel_weight"]["Social_Video"]
    t["channel_weight_rel"] = {ch: w / base_w for ch, w in truth["channel_weight"].items()}
    return t
