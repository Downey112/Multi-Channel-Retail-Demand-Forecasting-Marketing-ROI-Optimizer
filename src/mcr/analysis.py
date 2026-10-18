"""Estimators. Nothing here may import mcr.simulate: the analysis only sees what the database holds."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from .transforms import adbudg_inflection, adbudg_shape, adstock

PROMOS = ["Flash_Sale", "Bundle_Discount", "Loyalty_Cashback"]   # vs reference No_Promo
REFERENCE = "No_Promo"
CHANNELS = ["Social_Video", "Search_SEM", "In_App_Push", "Out_Of_Home"]   # first = weight 1 (numeraire)


# ---------------------------------------------------------------- frames (mirror the SQL views)

def store_week_frame(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Same rows and modelling columns as vw_store_week, built from the raw tables."""
    df = (t["weekly_sales"].merge(t["promo_assignments"], on=["store_id", "week_start"])
          .merge(t["stores"], on="store_id").merge(t["weeks"], on="week_start"))
    df["festive"] = (df["festival"] != "").astype(int)
    return df.sort_values(["store_id", "week_start"]).reset_index(drop=True)


def chain_week_frame(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Same rows and modelling columns as vw_chain_week (the view adds two window columns for DBeaver)."""
    sw = store_week_frame(t)
    g = sw.groupby("week_start")
    out = pd.DataFrame({
        "revenue_rm": g["revenue_rm"].sum(),
        "festive": g["festive"].first(),
        "cpi_index": g["cpi_index"].first(),
        **{f"n_{p.lower()}": g["promo_type"].apply(lambda s, p=p: int((s == p).sum())) for p in PROMOS},
    })
    spend = t["marketing_spend"].pivot(index="week_start", columns="channel", values="spend_k")[CHANNELS]
    spend.columns = [f"spend_{c.lower()}_k" for c in CHANNELS]
    out = out.join(spend).reset_index()
    out["total_spend_k"] = out[[f"spend_{c.lower()}_k" for c in CHANNELS]].sum(axis=1)
    return out


SPEND_COLS = [f"spend_{c.lower()}_k" for c in CHANNELS]
PROMO_COUNT_COLS = [f"n_{p.lower()}" for p in PROMOS]


# ---------------------------------------------------------------- module 1: promo experiment

def _fe_ols(df: pd.DataFrame, extra: pd.DataFrame):
    """OLS of store-week revenue on store and week fixed effects plus extra regressors."""
    X = pd.concat([
        pd.get_dummies(df["store_id"], prefix="s", dtype=float),
        pd.get_dummies(df["week_start"].astype(str), prefix="w", drop_first=True, dtype=float),
        extra,
    ], axis=1)
    return sm.OLS(df["revenue_rm"].to_numpy(float), X).fit()


def _f_test(full, restricted) -> tuple[float, float, float, float]:
    df_num = restricted.df_resid - full.df_resid
    ss = restricted.ssr - full.ssr
    f = (ss / df_num) / (full.ssr / full.df_resid)
    return ss, df_num, f, float(stats.f.sf(f, df_num, full.df_resid))


def promo_experiment(sw: pd.DataFrame) -> dict:
    """Split-plot analysis of the randomised promo experiment.

    Stores are whole plots: region is a between-store factor, tested on store means against
    store-within-region variation, with floor area as a store-level covariate (ANCOVA, 3 and 7 df).
    Store-weeks are subplots (promo and region x promo are tested against
    the residual, after store and week fixed effects absorb store levels and every chain-wide
    weekly effect such as media, season, festivals and common shocks).
    """
    n_weeks = sw["week_start"].nunique()
    # whole plot: ANCOVA of store means on region, adjusting for floor area (SS scaled to store-weeks)
    sm_ = sw.groupby(["store_id", "region"]).agg(rev=("revenue_rm", "mean"),
                                                 sqft=("floor_area_sqft", "first")).reset_index()
    X_cov = sm.add_constant(sm_[["sqft"]].astype(float))
    X_full = pd.concat([X_cov, pd.get_dummies(sm_["region"], drop_first=True, dtype=float)], axis=1)
    w_full, w_cov = sm.OLS(sm_["rev"], X_full).fit(), sm.OLS(sm_["rev"], X_cov).fit()
    ss_region = n_weeks * (w_cov.ssr - w_full.ssr)
    ss_store = n_weeks * w_full.ssr
    df_region, df_store = int(w_cov.df_resid - w_full.df_resid), int(w_full.df_resid)
    f_region = (ss_region / df_region) / (ss_store / df_store)

    dummies = pd.DataFrame({p: (sw["promo_type"] == p).astype(float) for p in PROMOS})
    cells = pd.DataFrame({f"{r}:{p}": ((sw["region"] == r) & (sw["promo_type"] == p)).astype(float)
                          for r in sorted(sw["region"].unique()) for p in PROMOS})
    m_blocks = _fe_ols(sw, pd.DataFrame(index=sw.index))
    m_add = _fe_ols(sw, dummies)
    m_int = _fe_ols(sw, cells)
    # both subplot effects are tested against the full subplot residual (interaction model)
    ss_p, df_p = m_blocks.ssr - m_add.ssr, m_blocks.df_resid - m_add.df_resid
    f_p = (ss_p / df_p) / m_int.scale
    p_p = float(stats.f.sf(f_p, df_p, m_int.df_resid))
    ss_i, df_i, f_i, p_i = _f_test(m_int, m_add)
    anova = [
        {"source": "Region (whole plot)", "ss": ss_region, "df": df_region, "f": f_region,
         "p": float(stats.f.sf(f_region, df_region, df_store)), "error_term": "Store(Region)"},
        {"source": "Store(Region) (whole-plot error)", "ss": ss_store, "df": df_store, "f": None, "p": None,
         "error_term": None},
        {"source": "Promo", "ss": ss_p, "df": df_p, "f": f_p, "p": p_p, "error_term": "Residual"},
        {"source": "Region x Promo", "ss": ss_i, "df": df_i, "f": f_i, "p": p_i, "error_term": "Residual"},
        {"source": "Residual (subplot error)", "ss": m_int.ssr, "df": m_int.df_resid, "f": None, "p": None,
         "error_term": None},
    ]
    for row in anova:
        row["partial_eta_sq"] = (row["ss"] / (row["ss"] + (ss_store if row["error_term"] == "Store(Region)"
                                 else m_int.ssr))) if row["f"] is not None else None

    # Marginal promo effects = equal-weight average of the region cells of the interaction model, so the
    # contrasts use the full subplot error. Tukey-Kramer on these (correlated) contrasts is the usual
    # approximation; it is conservative (Hayter 1984).
    regions = sorted(sw["region"].unique())
    levels = PROMOS + [REFERENCE]
    L = pd.DataFrame(0.0, index=levels, columns=m_int.params.index)
    for p in PROMOS:
        L.loc[p, [f"{r}:{p}" for r in regions]] = 1.0 / len(regions)
    b = L @ m_int.params
    V = L @ m_int.cov_params() @ L.T
    q_crit = float(stats.studentized_range.ppf(0.95, len(levels), m_int.df_resid))
    t_crit = float(stats.t.ppf(0.975, m_int.df_resid))
    tukey = []
    for i, li in enumerate(levels):
        for lj in levels[i + 1:]:
            se = float(np.sqrt(V.loc[li, li] + V.loc[lj, lj] - 2 * V.loc[li, lj]))
            d = float(b[li] - b[lj])
            q = abs(d) / se * np.sqrt(2)
            tukey.append({"comparison": f"{li} vs {lj}", "diff_rm": d, "se": se,
                          "ci_low": d - q_crit / np.sqrt(2) * se, "ci_high": d + q_crit / np.sqrt(2) * se,
                          "q": q, "p_adj": float(stats.studentized_range.sf(q, len(levels), m_int.df_resid))})

    # simple effects: lift vs No_Promo within each region (interaction model)
    ci = m_int.conf_int()
    simple = [{"region": k.split(":")[0], "promo": k.split(":")[1], "lift_rm": float(m_int.params[k]),
               "ci_low": float(ci.loc[k, 0]), "ci_high": float(ci.loc[k, 1])} for k in cells.columns]

    # residual checks: within-store autocorrelation and equal variance across promos
    res = pd.Series(m_int.resid, index=sw.index)
    ac = sw.assign(r=res).groupby("store_id")["r"].apply(lambda r: r.autocorr(1)).mean()
    lev = stats.levene(*[res[sw["promo_type"] == p] for p in levels])
    return {
        "n_store_weeks": len(sw), "anova": anova,
        "promo_lift": [{"promo": p, "lift_rm": float(b[p]),
                        "ci_low": float(b[p] - t_crit * np.sqrt(V.loc[p, p])),
                        "ci_high": float(b[p] + t_crit * np.sqrt(V.loc[p, p]))} for p in PROMOS],
        "tukey": tukey, "simple_effects": simple,
        "residual_sd_rm": float(np.sqrt(m_int.scale)),
        "mean_within_store_lag1_autocorr": float(ac),
        "levene_p": float(lev.pvalue),
    }


# ---------------------------------------------------------------- module 2: chain-level regression

def _controls(ch: pd.DataFrame, cpi_base: float | None = None) -> pd.DataFrame:
    t = np.arange(len(ch))
    base = ch["cpi_index"].iloc[0] if cpi_base is None else cpi_base
    return pd.DataFrame({
        "const": 1.0, "festive": ch["festive"].to_numpy(float),
        "cpi_dev": ch["cpi_index"].to_numpy(float) - base,
        "season_sin": np.sin(2 * np.pi * t / 52), "season_cos": np.cos(2 * np.pi * t / 52),
        **{c: ch[c].to_numpy(float) for c in PROMO_COUNT_COLS},
    }, index=ch.index)


def _aic(sse: float, n: int, k: int) -> float:
    """Gaussian AIC up to a constant: n log(SSE / n) + 2k. Only differences between models matter."""
    return float(n * np.log(sse / n) + 2 * k)


def linear_media_ols(ch: pd.DataFrame) -> dict:
    """Linear distributed-lag model (current + last week's total spend) with Newey-West errors."""
    X = _controls(ch)
    X["spend_k"] = ch["total_spend_k"].to_numpy(float)
    X["spend_lag1_k"] = X["spend_k"].shift(1)
    X, y = X.iloc[1:], ch["revenue_rm"].iloc[1:].to_numpy(float)
    ols = sm.OLS(y, X).fit()
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    vif = {c: float(variance_inflation_factor(X.to_numpy(), i)) for i, c in enumerate(X.columns) if c != "const"}
    bp = het_breuschpagan(ols.resid, X)
    bg = acorr_breusch_godfrey(ols, nlags=4, result_object=False)
    # RESET with standardised fitted values (raw powers of ~1e6 revenues are numerically singular)
    z = (ols.fittedvalues - ols.fittedvalues.mean()) / ols.fittedvalues.std()
    # and a HAC Wald test of the two added terms, since the errors are autocorrelated
    aux = sm.OLS(y, np.column_stack([X, z ** 2, z ** 3])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    R = np.zeros((2, len(aux.params)))
    R[0, -2] = R[1, -1] = 1.0
    reset = aux.wald_test(R, scalar=True, use_f=True)
    reset_f, reset_p = float(reset.statistic), float(reset.pvalue)
    jb = jarque_bera(ols.resid)
    return {
        "n": int(ols.nobs), "r2": ols.rsquared, "adj_r2": ols.rsquared_adj,
        "aic": _aic(ols.ssr, int(ols.nobs), len(X.columns)),   # same formula as the NLS fit
        "coef": [{"term": c, "estimate": float(hac.params[c]), "hac_se": float(hac.bse[c]),
                  "ols_se": float(ols.bse[c]), "p_hac": float(hac.pvalues[c]),
                  "vif": vif.get(c)} for c in X.columns],
        "diagnostics": {
            "breusch_pagan_p": float(bp[1]), "breusch_godfrey_lm4_p": float(bg[1]),
            "reset_hac_f": reset_f, "reset_hac_p": reset_p,
            "jarque_bera_p": float(jb[1]), "shapiro_p": float(stats.shapiro(ols.resid).pvalue),
            "durbin_watson": float(durbin_watson(ols.resid)),
        },
    }


# ---------------------------------------------------------------- module 2b: ADBUDG media response

THETA_NAMES = ["adstock_lambda", "c", "half_k", "w_search_sem", "w_in_app_push", "w_out_of_home"]
THETA_LO = np.array([0.0, 0.5, 2.0, 0.05, 0.05, 0.05])
THETA_HI = np.array([0.95, 5.0, 300.0, 5.0, 5.0, 5.0])


def _media_regressor(spend: np.ndarray, theta: np.ndarray) -> np.ndarray:
    lam, c, half, *w = theta
    eff = spend @ np.r_[1.0, w]
    return adbudg_shape(adstock(eff, lam), c, half)


def _vp_fit(y, Z, spend, theta):
    """Variable projection: given the nonlinear parameters, the linear ones (incl. ceiling) are OLS."""
    X = np.column_stack([Z, _media_regressor(spend, theta)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def fit_response(ch: pd.DataFrame, starts: list[np.ndarray] | None = None, cpi_base: float | None = None) -> dict:
    """Chain revenue = controls + ceiling * ADBUDG(adstock(sum_k w_k spend_k)), fitted jointly by NLS."""
    y = ch["revenue_rm"].to_numpy(float)
    Z = _controls(ch, cpi_base)
    spend = ch[SPEND_COLS].to_numpy(float)
    if starts is None:
        starts = [np.array([lam, c, h, 1.0, 1.0, 1.0]) for lam in (0.2, 0.5, 0.8) for c in (1.2, 2.5)
                  for h in (20.0, 60.0)]
    best = None
    for s in starts:
        r = optimize.least_squares(lambda th: _vp_fit(y, Z.to_numpy(), spend, th)[1], s,
                                   bounds=(THETA_LO, THETA_HI), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    theta = best.x
    beta, resid = _vp_fit(y, Z.to_numpy(), spend, theta)
    n, k = len(y), len(beta) + len(theta)
    sse = float(resid @ resid)
    return {
        "theta": dict(zip(THETA_NAMES, map(float, theta))),
        "beta": dict(zip(list(Z.columns) + ["ceiling_rm"], map(float, beta))),
        "sse": sse, "n": n, "k": k, "aic": _aic(sse, n, k),
        # on weeks 2..n, the sample the distributed-lag OLS can use
        "aic_from_week2": _aic(float(resid[1:] @ resid[1:]), n - 1, k),
        "r2": 1 - sse / float(((y - y.mean()) ** 2).sum()),
        "fitted": (y - resid).tolist(), "resid": resid.tolist(),
    }


def response_params(fit: dict) -> dict:
    """The reported parameters of a response fit: media parameters plus festive and CPI effects."""
    return {**fit["theta"], "ceiling_rm": fit["beta"]["ceiling_rm"], "festive_rm": fit["beta"]["festive"],
            "cpi_rm_per_point": fit["beta"]["cpi_dev"]}


def bootstrap_response(ch: pd.DataFrame, fit: dict, n_boot: int = 300, block: int = 8, seed: int = 7) -> pd.DataFrame:
    """Moving-block residual bootstrap (keeps the residual autocorrelation); refits the full NLS each time."""
    rng = np.random.default_rng(seed)
    fitted, resid = np.array(fit["fitted"]), np.array(fit["resid"])
    n = len(resid)
    theta0 = np.array(list(fit["theta"].values()))
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block + 1, n // block + 1)])[:n]
        chb = ch.copy()
        e = resid[idx]
        chb["revenue_rm"] = fitted + e - e.mean()                    # centred resampled residuals
        far = np.array([0.5, 1.2, 150.0, 1.0, 1.0, 1.0])            # a start far from theta0 (flat curve)
        f = fit_response(chb, starts=[theta0, np.clip(theta0 * rng.uniform(0.7, 1.3, 6), THETA_LO, THETA_HI), far])
        rows.append(response_params(f))
    return pd.DataFrame(rows)


def mix_weight(theta: dict, mix: dict[str, float]) -> float:
    """Social-equivalent weight of one RM in the given channel mix (shares sum to 1)."""
    w = {"Social_Video": 1.0, "Search_SEM": theta["w_search_sem"], "In_App_Push": theta["w_in_app_push"],
         "Out_Of_Home": theta["w_out_of_home"]}
    return sum(mix[c] * w[c] for c in CHANNELS)


def long_run_lift(spend_k: np.ndarray, theta: dict, ceiling: float, mix: dict[str, float]) -> np.ndarray:
    """Steady-state weekly chain revenue lift from a constant weekly budget (RM '000) with this mix."""
    a = np.asarray(spend_k, float) * mix_weight(theta, mix) / (1 - theta["adstock_lambda"])
    return ceiling * adbudg_shape(a, theta["c"], theta["half_k"])


def roi_summary(theta: dict, ceiling: float, mix: dict[str, float], margin: float) -> dict:
    """Inflection, half-saturation and profit-optimal weekly budget for a fixed channel mix."""
    to_spend = (1 - theta["adstock_lambda"]) / mix_weight(theta, mix)      # adstock units -> RM '000 / week
    infl = adbudg_inflection(theta["c"], theta["half_k"]) * to_spend
    half = theta["half_k"] * to_spend

    def marginal(s):  # RM of gross profit per extra RM of spend, minus 1
        h = 1e-4
        return margin * (long_run_lift(s + h, theta, ceiling, mix) - long_run_lift(s - h, theta, ceiling, mix)) \
            / (2 * h) / 1000.0 - 1.0

    hi = max(infl, 1.0)
    while marginal(hi) > 0 and hi < 1e4:
        hi *= 1.5
    lo = max(infl, 1e-3)
    # past the inflection the marginal return only falls, so a root there is the interior maximum;
    # it must also beat spending nothing (an S-curve can have a root where total profit is negative)
    s_opt = float(optimize.brentq(marginal, lo, hi)) if marginal(lo) > 0 > marginal(hi) else 0.0
    profit = lambda s: margin * float(long_run_lift(s, theta, ceiling, mix)) - 1000.0 * s  # noqa: E731
    if s_opt > 0 and profit(s_opt) <= 0:
        s_opt = 0.0
    return {"inflection_spend_k": float(infl), "half_saturation_spend_k": float(half),
            "optimal_spend_k": s_opt, "weekly_profit_at_optimum_rm": profit(s_opt), "margin_assumed": margin}


def channel_marginal_roi(theta: dict, ceiling: float, spend_k: dict[str, float]) -> dict[str, float]:
    """Long-run revenue from one extra RM 1,000 per week in each channel, at the given weekly spend."""
    w = {"Social_Video": 1.0, "Search_SEM": theta["w_search_sem"], "In_App_Push": theta["w_in_app_push"],
         "Out_Of_Home": theta["w_out_of_home"]}
    lam = theta["adstock_lambda"]
    a = sum(w[c] * spend_k[c] for c in CHANNELS) / (1 - lam)
    h = 1e-4
    dg = (adbudg_shape(a + h, theta["c"], theta["half_k"]) - adbudg_shape(a - h, theta["c"], theta["half_k"])) / (2 * h)
    return {c: float(ceiling * dg * w[c] / (1 - lam)) for c in CHANNELS}


# ---------------------------------------------------------------- module 3: forecasting

CPI_LAG_WEEKS = 8      # a month's CPI is known ~4 weeks after month end: up to 8 weeks after a week starts

def forecast_design(ch: pd.DataFrame, theta: dict, k_fourier: int, cpi_known_through: int | None = None) -> pd.DataFrame:
    """Exogenous regressors for chain revenue.

    CPI enters with a CPI_LAG_WEEKS lag (DOSM publishes a month's CPI about four weeks after it ends), and
    rows after cpi_known_through carry the last available value forward.
    """
    t = np.arange(len(ch))
    raw = ch["cpi_index"].to_numpy(float)
    cpi = np.r_[np.full(CPI_LAG_WEEKS, raw[0]), raw[:-CPI_LAG_WEEKS]]
    if cpi_known_through is not None:
        cpi[cpi_known_through + 1:] = cpi[cpi_known_through]
    X = pd.DataFrame({"const": 1.0, "festive": ch["festive"].to_numpy(float), "cpi_dev": cpi - cpi[0]},
                     index=ch.index)
    for k in range(1, k_fourier + 1):
        X[f"sin{k}"] = np.sin(2 * np.pi * k * t / 52)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * t / 52)
    for c in PROMO_COUNT_COLS:
        X[c] = ch[c].to_numpy(float)
    X["media"] = _media_regressor(ch[SPEND_COLS].to_numpy(float), np.array(list(theta.values())))
    return X


def stationarity(y: np.ndarray) -> dict:
    from statsmodels.tsa.stattools import adfuller, kpss
    import warnings
    out = {}
    for name, series in (("level", y), ("first_difference", np.diff(y))):
        adf = adfuller(series, autolag="AIC", regression="c", result_object=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # kpss warns when p is outside its lookup table
            kp = kpss(series, regression="c", nlags="auto")
        out[name] = {"adf_stat": float(adf[0]), "adf_p": float(adf[1]), "kpss_stat": float(kp[0]),
                     "kpss_p": float(kp[1])}
    return out


def select_sarimax(y: np.ndarray, X: pd.DataFrame, max_pq: int = 2):
    """Regression with ARMA(p, q) errors; (p, q) by AIC. Returns (aic table, best fitted result)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings
    rows, best = [], None
    for p in range(max_pq + 1):
        for q in range(max_pq + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = SARIMAX(y, exog=X.to_numpy(), order=(p, 0, q)).fit(disp=False)
            rows.append({"p": p, "q": q, "aic": float(r.aic)})
            if best is None or r.aic < best.aic:
                best = r
    return pd.DataFrame(rows), best


def accuracy(actual: np.ndarray, pred: np.ndarray) -> dict:
    e = actual - pred
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "mape_pct": float(np.mean(np.abs(e / actual)) * 100)}
