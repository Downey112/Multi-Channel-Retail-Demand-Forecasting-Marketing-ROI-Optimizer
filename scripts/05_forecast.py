"""Module 3 (STA570): stationarity tests, regression with ARMA errors, 13-week holdout and forecast.

Model: chain revenue = festive + CPI + Fourier annual terms + promo counts + media(ADBUDG of adstock)
+ ARMA(p, q) errors, fitted with statsmodels SARIMAX. There is no seasonal ARIMA lag: annual seasonality
on weekly data is carried by Fourier terms (a dynamic harmonic regression).

Information rules for the holdout and the forecast: the festive calendar, the promo calendar and the
media plan are known in advance; CPI is not (it is published after the month ends), so it is carried
forward from the last training week. Media parameters are re-estimated on the training window only.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcr import plotstyle as ps, report  # noqa: E402
from mcr.analysis import (CHANNELS, PROMO_COUNT_COLS, SPEND_COLS, accuracy, fit_response,  # noqa: E402
                          forecast_design, select_sarimax, stationarity)
from mcr.calendar import festive_weeks  # noqa: E402
from mcr.db import connect, engine  # noqa: E402

HOLDOUT = 13
HORIZON = 13
N_PATHS = 2000


def load_chain() -> pd.DataFrame:
    ch = pd.read_sql("SELECT * FROM vw_chain_week ORDER BY week_start", engine())
    num = [c for c in ch.columns if c != "week_start"]
    ch[num] = ch[num].astype(float)
    ch["week_start"] = pd.to_datetime(ch["week_start"])
    return ch


def choose_fourier(y, ch, theta, known_through) -> tuple[int, pd.DataFrame]:
    """Number of Fourier pairs by AIC of the exogenous OLS (K = 1..3)."""
    aics = {}
    for k in (1, 2, 3):
        X = forecast_design(ch, theta, k, known_through).iloc[:len(y)]
        aics[k] = sm.OLS(y, X).fit().aic
    k = min(aics, key=aics.get)
    return k, pd.DataFrame({"k": list(aics), "aic": list(aics.values())})


def future_plan(ch: pd.DataFrame, total_k: float | None) -> pd.DataFrame:
    """Next HORIZON weeks: real festive calendar, balanced promo calendar, a media plan.

    The plan repeats the last 52 weeks' average spend by channel, split by festive / normal weeks;
    if total_k is given, normal-week spend is rescaled to that weekly total (same mix). Festive weeks
    keep their usual spend: the module 2 optimum is a steady weekly budget, and festive weeks already
    run high on the curve.
    """
    last = ch["week_start"].iloc[-1]
    wk = pd.date_range(last + pd.Timedelta(weeks=1), periods=HORIZON, freq="W-MON")
    fest = (festive_weeks(wk) != "").astype(int).to_numpy()
    recent = ch.iloc[-52:]
    recent = recent[recent["total_spend_k"] > 0]
    normal = recent[recent["festive"] == 0][SPEND_COLS].mean()
    festive = recent[recent["festive"] == 1][SPEND_COLS].mean()
    if total_k is not None:
        normal = normal * total_k / normal.sum()
    fut = pd.DataFrame({"week_start": wk, "festive": fest, "cpi_index": ch["cpi_index"].iloc[-1],
                        **{c: 3.0 for c in PROMO_COUNT_COLS}})
    for c in SPEND_COLS:
        fut[c] = np.where(fest == 1, festive[c], normal[c])
    fut["total_spend_k"] = fut[SPEND_COLS].sum(axis=1)
    return fut


def main() -> None:
    ch = load_chain()
    y = ch["revenue_rm"].to_numpy()
    n_train = len(ch) - HOLDOUT
    train = ch.iloc[:n_train]
    station = stationarity(train["revenue_rm"].to_numpy())
    print("stationarity (training window):")
    for k, v in station.items():
        print(f"  {k:<17} ADF {v['adf_stat']:7.3f} (p {v['adf_p']:.3f})   KPSS {v['kpss_stat']:.3f} (p {v['kpss_p']:.3f})")

    # --- holdout: everything estimated on the first n_train weeks
    theta_tr = fit_response(train)["theta"]
    k_four, four_tab = choose_fourier(train["revenue_rm"].to_numpy(), ch, theta_tr, n_train - 1)
    X = forecast_design(ch, theta_tr, k_four, cpi_known_through=n_train - 1)
    resid_ols = sm.OLS(y[:n_train], X.iloc[:n_train]).fit()
    aic_tab, best = select_sarimax(y[:n_train], X.iloc[:n_train])
    p, q = best.model.order[0], best.model.order[2]
    print(f"Fourier pairs K = {k_four}; ARMA errors chosen by AIC: ({p}, {q})")
    pred = best.get_forecast(HOLDOUT, exog=X.iloc[n_train:].to_numpy())
    test = y[n_train:]
    sarimax_pred = pred.predicted_mean
    ols_pred = resid_ols.predict(X.iloc[n_train:])
    naive = y[n_train - 52:len(y) - 52]
    ci_h = pred.conf_int(alpha=0.05)
    cover = float(np.mean((test >= ci_h[:, 0]) & (test <= ci_h[:, 1])))
    holdout = {"sarimax": accuracy(test, sarimax_pred), "regression_no_arma": accuracy(test, ols_pred.to_numpy()),
               "seasonal_naive": accuracy(test, naive), "sarimax_95pi_coverage": cover}
    for k, v in holdout.items():
        print(f"  {k:<20} {v}")

    # --- refit on all weeks and forecast two media plans
    fit_full = fit_response(ch)
    theta = fit_full["theta"]
    resp = report.load("response")
    plans = {"current_plan": None, "profit_optimal_budget": resp["roi"]["optimal_spend_k"]}
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    X_hist = forecast_design(ch, theta, k_four)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y, exog=X_hist.to_numpy(), order=(p, 0, q)).fit(disp=False)
    out, paths_total = {}, {}
    for name, total in plans.items():
        fut = future_plan(ch, total)
        allx = pd.concat([ch, fut], ignore_index=True)
        Xa = forecast_design(allx, theta, k_four, cpi_known_through=len(ch) - 1)
        assert np.allclose(Xa.iloc[:len(ch)].to_numpy(), X_hist.to_numpy())
        fc = res.get_forecast(HORIZON, exog=Xa.iloc[len(ch):].to_numpy())
        ci_f = fc.conf_int(alpha=0.05)
        sims = np.squeeze(res.simulate(HORIZON, repetitions=N_PATHS, anchor="end",
                                       exog=Xa.iloc[len(ch):].to_numpy(), rng=np.random.default_rng(11)))
        tot = sims.sum(axis=0)
        paths_total[name] = tot
        out[name] = {
            "weekly": [{"week_start": str(w.date()), "planned_spend_k": float(s), "point_rm": float(m),
                        "lo95_rm": float(lo), "hi95_rm": float(hi)}
                       for w, s, m, lo, hi in zip(fut["week_start"], fut["total_spend_k"], fc.predicted_mean,
                                                  ci_f[:, 0], ci_f[:, 1])],
            "total_13w_rm": float(fc.predicted_mean.sum()),
            "total_13w_pi95_rm": [float(np.quantile(tot, 0.025)), float(np.quantile(tot, 0.975))],
            "media_spend_13w_k": float(fut["total_spend_k"].sum()),
        }
        print(f"  {name:<22} 13-week revenue RM {out[name]['total_13w_rm']:,.0f}  95% PI "
              f"[{out[name]['total_13w_pi95_rm'][0]:,.0f}, {out[name]['total_13w_pi95_rm'][1]:,.0f}]  "
              f"media RM {out[name]['media_spend_13w_k']:.0f}k")
    diff = paths_total["profit_optimal_budget"] - paths_total["current_plan"]
    extra_media = 1000 * (out["profit_optimal_budget"]["media_spend_13w_k"] - out["current_plan"]["media_spend_13w_k"])
    margin = resp["roi"]["margin_assumed"]
    net = margin * diff - extra_media
    print(f"  optimal - current, same shocks: revenue RM {np.mean(diff):,.0f}, extra media RM {extra_media:,.0f}, "
          f"net gross profit at {margin:.0%} margin RM {np.mean(net):,.0f}")

    report.save("forecast", {
        "model": f"Regression with ARMA({p},{q}) errors, {k_four} Fourier pair(s), statsmodels SARIMAX",
        "stationarity": station, "fourier_aic": four_tab.to_dict("records"), "arma_aic": aic_tab.to_dict("records"),
        "holdout_weeks": HOLDOUT, "holdout_start": str(ch["week_start"].iloc[n_train].date()), "holdout": holdout,
        "holdout_series": {"week_start": [str(d.date()) for d in ch["week_start"].iloc[n_train:]],
                           "actual": test.tolist(), "sarimax": sarimax_pred.tolist(), "lo95": ci_h[:, 0].tolist(),
                           "hi95": ci_h[:, 1].tolist(), "seasonal_naive": naive.tolist()},
        "forecast": out, "plan_difference_mean_rm": float(np.mean(diff)), "plan_extra_media_rm": extra_media,
        "plan_net_profit_mean_rm": float(np.mean(net)), "margin_assumed": margin,
        "information_rules": "festive calendar, promo calendar and media plan known in advance; CPI carried forward",
    })

    with connect() as conn:
        run_id = conn.execute(
            "INSERT INTO model_runs (model, spec, trained_through) VALUES (%s, %s, %s) RETURNING run_id",
            ("sarimax_regression_arma", f"ARMA({p},{q}) errors + {k_four} Fourier + festive + CPI + promo + ADBUDG media",
             ch["week_start"].iloc[-1].date())).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO forecasts VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(run_id, name, r["week_start"], round(r["planned_spend_k"], 3), round(r["point_rm"], 2),
                  round(r["lo95_rm"], 2), round(r["hi95_rm"], 2)) for name, o in out.items() for r in o["weekly"]])
    print(f"  wrote run {run_id} ({2 * HORIZON} rows) to forecasts")

    hs = report.load("forecast")["holdout_series"]
    fig, ax = ps.plt.subplots(figsize=(9, 4.4))
    hist = ch.iloc[-52:]
    ax.plot(hist["week_start"], hist["revenue_rm"] / 1e6, color=ps.INK2, linewidth=1.4, label="actual")
    hw = pd.to_datetime(hs["week_start"])
    ax.plot(hw, np.array(hs["sarimax"]) / 1e6, color=ps.SERIES[0], label="holdout forecast")
    ax.fill_between(hw, np.array(hs["lo95"]) / 1e6, np.array(hs["hi95"]) / 1e6, color=ps.SEQ[1], alpha=0.5,
                    linewidth=0)
    ax.plot(hw, np.array(hs["seasonal_naive"]) / 1e6, color=ps.MUTED, linestyle=":", label="seasonal naive")
    fw = pd.to_datetime([r["week_start"] for r in out["current_plan"]["weekly"]])
    cur = out["current_plan"]["weekly"]
    ax.plot(fw, [r["point_rm"] / 1e6 for r in cur], color=ps.SERIES[1], label="forecast, current plan")
    ax.fill_between(fw, [r["lo95_rm"] / 1e6 for r in cur], [r["hi95_rm"] / 1e6 for r in cur],
                    color=ps.SERIES[1], alpha=0.15, linewidth=0)
    ax.set_ylabel("weekly chain revenue (RM million)")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    ax.set_title("13-week holdout and 13-week forecast (95% intervals)")
    ps.save(fig, "forecast.png")


if __name__ == "__main__":
    main()
