"""Streamlit ROI simulator. Reads the saved results (docs/results/*.json); run the pipeline first.

    streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from mcr import report  # noqa: E402
from mcr.analysis import CHANNELS, channel_marginal_roi, long_run_lift, roi_summary  # noqa: E402

st.set_page_config(page_title="Media ROI Simulator", layout="wide")
resp, exp, fc = report.load("response"), report.load("experiment"), report.load("forecast")
theta = resp["adbudg"]["theta"]
ceiling = resp["adbudg"]["beta"]["ceiling_rm"]
draws = pd.DataFrame(resp["bootstrap_draws"])

st.title("Multi-channel media ROI simulator")
st.caption("Simulation study: store sales, promotions and media spend are simulated with known true parameters; "
           "the festive calendar and CPI are real. Estimates come from the fitted models, not the simulator.")

left, right = st.columns([1, 2])
with left:
    budget = st.slider("Weekly media budget (RM '000)", 0.0, 120.0, float(round(resp["current_budget_k"], 1)), 0.5)
    margin = st.slider("Gross margin (assumption)", 0.10, 0.60, float(resp["roi"]["margin_assumed"]), 0.01)
    st.markdown("**Channel mix** (normalised to 100%)")
    raw = {c: st.slider(c.replace("_", " "), 0.0, 1.0, float(round(resp["channel_mix_last52"][c], 2)), 0.01)
           for c in CHANNELS}
    tot = sum(raw.values()) or 1.0
    mix = {c: v / tot for c, v in raw.items()}

th_cols = list(theta)
lift = float(long_run_lift(budget, theta, ceiling, mix))
band = np.array([long_run_lift(budget, {k: r[k] for k in th_cols}, r["ceiling_rm"], mix) for _, r in draws.iterrows()])
roi = roi_summary(theta, ceiling, mix, margin)
with right:
    a, b, c = st.columns(3)
    a.metric("Weekly revenue lift", f"RM {lift:,.0f}", help="Long-run (steady-state) lift vs no media")
    b.metric("95% bootstrap range", f"RM {np.quantile(band, 0.025):,.0f} – {np.quantile(band, 0.975):,.0f}")
    c.metric("Profit-optimal budget", f"RM {roi['optimal_spend_k']:.1f}k / week",
             help=f"Where an extra RM 1 of spend returns RM 1 of gross profit at {margin:.0%} margin")
    grid = np.arange(0, 121, 2.0)
    curves = np.array([long_run_lift(grid, {k: r[k] for k in th_cols}, r["ceiling_rm"], mix) for _, r in draws.iterrows()])
    chart = pd.DataFrame({"fitted": long_run_lift(grid, theta, ceiling, mix),
                          "lower 95%": np.quantile(curves, 0.025, axis=0),
                          "upper 95%": np.quantile(curves, 0.975, axis=0)}, index=grid)
    chart.index.name = "weekly budget (RM '000)"
    st.line_chart(chart, y_label="weekly revenue lift (RM)")
    spend_now = {ch: budget * mix[ch] for ch in CHANNELS}
    m = channel_marginal_roi(theta, ceiling, spend_now)
    st.markdown("**Long-run revenue from one extra RM 1,000 per week, by channel**")
    st.bar_chart(pd.Series({k.replace("_", " "): v for k, v in m.items()}), y_label="RM")
    st.caption("The response curve's ceiling and steepness are only weakly identified: past spend rarely went "
               "above RM 50k a week, so read budgets far above that with caution (see the bootstrap range).")

st.subheader("Promo experiment: lift vs no promo (RM per store-week, Tukey-Kramer 95%)")
st.dataframe(pd.DataFrame(exp["tukey"])[["comparison", "diff_rm", "ci_low", "ci_high", "p_adj"]].round(1),
             hide_index=True)

st.subheader("13-week forecast")
rows = [{"week": r["week_start"], "plan": name, "forecast": r["point_rm"], "lo95": r["lo95_rm"], "hi95": r["hi95_rm"]}
        for name, o in fc["forecast"].items() for r in o["weekly"]]
f = pd.DataFrame(rows)
st.line_chart(f.pivot(index="week", columns="plan", values="forecast"), y_label="weekly chain revenue (RM)")
st.caption(fc["model"] + ". " + fc["information_rules"] + ".")
