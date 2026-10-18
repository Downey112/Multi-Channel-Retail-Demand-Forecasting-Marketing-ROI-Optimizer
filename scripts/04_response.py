"""Module 2 (STA600 regression, STA589 marketing models): chain-level media response and ROI.

n = 156 chain-weeks. Media spend is decided at chain level, so the response is estimated on chain
totals, not on 1,872 store-weeks (which would repeat each week's spend 12 times).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcr import plotstyle as ps, report  # noqa: E402
from mcr.analysis import (CHANNELS, SPEND_COLS, bootstrap_response, channel_marginal_roi,  # noqa: E402
                          fit_response, linear_media_ols, long_run_lift, response_params, roi_summary)
from mcr.db import engine  # noqa: E402

MARGIN = 0.35        # ASSUMED gross margin, used to turn revenue into profit for the optimal budget
N_BOOT = 300


def load_chain() -> pd.DataFrame:
    ch = pd.read_sql("SELECT * FROM vw_chain_week ORDER BY week_start", engine())
    num = [c for c in ch.columns if c != "week_start"]
    ch[num] = ch[num].astype(float)
    return ch


def main() -> None:
    ch = load_chain()
    lin = linear_media_ols(ch)
    print(f"linear distributed-lag OLS: n {lin['n']}, adj R2 {lin['adj_r2']:.3f}, AIC {lin['aic']:.1f}")
    for c in lin["coef"]:
        vif = f"  VIF {c['vif']:.2f}" if c["vif"] else ""
        print(f"  {c['term']:<20} {c['estimate']:>12,.1f}  HAC se {c['hac_se']:>10,.1f}  p {c['p_hac']:.3g}{vif}")
    print("  diagnostics:", {k: round(v, 3) for k, v in lin["diagnostics"].items()})

    fit = fit_response(ch)
    print(f"ADBUDG NLS: R2 {fit['r2']:.4f}, AIC on weeks 2-156 {fit['aic_from_week2']:.1f} "
          f"(linear distributed lag {lin['aic']:.1f}; lower is better)")
    boot = bootstrap_response(ch, fit, n_boot=N_BOOT)
    params = response_params(fit)
    ci = {k: [float(boot[k].quantile(0.025)), float(boot[k].quantile(0.975))] for k in params}
    for k, v in params.items():
        print(f"  {k:<16} {v:>12,.3f}  95% CI [{ci[k][0]:,.3f}, {ci[k][1]:,.3f}]")

    recent = ch.iloc[-52:]
    mix = {c: float(recent[s].sum() / recent["total_spend_k"].sum()) for c, s in zip(CHANNELS, SPEND_COLS)}
    roi = roi_summary(fit["theta"], params["ceiling_rm"], mix, MARGIN)
    roi_boot = [roi_summary({k: r[k] for k in fit["theta"]}, r["ceiling_rm"], mix, MARGIN)
                for _, r in boot.iterrows()]
    roi_ci = {k: [float(np.nanquantile([b[k] for b in roi_boot], q)) for q in (0.025, 0.975)]
              for k in ("inflection_spend_k", "half_saturation_spend_k", "optimal_spend_k")}
    nonzero = recent[recent["total_spend_k"] > 0]
    current = float(nonzero["total_spend_k"].mean())
    ch_spend = {c: float(nonzero[s].mean()) for c, s in zip(CHANNELS, SPEND_COLS)}
    mroi = channel_marginal_roi(fit["theta"], params["ceiling_rm"], ch_spend)
    print("channel mix (last 52 weeks):", {k: round(v, 3) for k, v in mix.items()})
    print(f"current average budget RM {current:.1f}k/week; inflection RM {roi['inflection_spend_k']:.1f}k "
          f"{roi_ci['inflection_spend_k']}; profit-optimal at {MARGIN:.0%} margin RM {roi['optimal_spend_k']:.1f}k "
          f"{roi_ci['optimal_spend_k']}")
    print("long-run revenue per extra RM 1,000/week:", {k: round(v) for k, v in mroi.items()})

    grid = np.arange(0, 121, 5.0)
    curve = long_run_lift(grid, fit["theta"], params["ceiling_rm"], mix)
    band = np.array([long_run_lift(grid, {k: r[k] for k in fit["theta"]}, r["ceiling_rm"], mix)
                     for _, r in boot.iterrows()])
    lo, hi = np.quantile(band, 0.025, axis=0), np.quantile(band, 0.975, axis=0)
    report.save("response", {
        "linear_ols": lin, "adbudg": {k: v for k, v in fit.items() if k not in ("fitted", "resid")},
        "adbudg_ci95": ci, "n_boot": N_BOOT, "bootstrap": "moving-block residual bootstrap, block 8 weeks",
        "bootstrap_draws": boot.round(6).to_dict("records"),
        "channel_mix_last52": mix, "current_budget_k": current, "channel_spend_last52_k": ch_spend,
        "roi": roi, "roi_ci95": roi_ci, "channel_marginal_revenue_per_1k": mroi,
        "curve": {"spend_k": grid.tolist(), "lift_rm": curve.tolist(), "lo95": lo.tolist(), "hi95": hi.tolist()},
    })

    fig, ax = ps.plt.subplots(figsize=(7.4, 4.4))
    ax.fill_between(grid, lo / 1e3, hi / 1e3, color=ps.SEQ[1], alpha=0.6, linewidth=0, label="95% bootstrap band")
    ax.plot(grid, curve / 1e3, color=ps.SERIES[0], label="fitted ADBUDG (long run)")
    for x, lab, col in ((roi["inflection_spend_k"], "inflection", ps.MUTED), (current, "current budget", ps.INK2),
                        (roi["optimal_spend_k"], f"profit-optimal ({MARGIN:.0%} margin)", ps.SERIES[1])):
        ax.axvline(x, color=col, linewidth=1, linestyle="--")
        ax.text(x + 1, ax.get_ylim()[1] * 0.04, lab, rotation=90, fontsize=8, color=col, va="bottom")
    ax.set_xlabel("weekly media budget, current channel mix (RM '000)")
    ax.set_ylabel("weekly chain revenue lift (RM '000)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Media response curve: diminishing returns past the inflection")
    ps.save(fig, "media_response_curve.png")

    fig, ax = ps.plt.subplots(figsize=(7.4, 3.2))
    names = [c.replace("_", " ") for c in CHANNELS]
    ax.barh(names[::-1], [mroi[c] / 1e3 for c in CHANNELS][::-1], color=ps.SERIES[0], height=0.55)
    ax.axvline(1 / MARGIN, color=ps.SERIES[1], linewidth=1, linestyle="--")
    ax.text(1 / MARGIN, 3.35, f" break-even at {MARGIN:.0%} margin", fontsize=8, color=ps.SERIES[1])
    ax.set_xlabel("long-run revenue per extra RM 1 of weekly spend")
    ax.grid(axis="y", visible=False)
    ax.set_title("Marginal return by channel at current spend")
    ps.save(fig, "channel_marginal_return.png")


if __name__ == "__main__":
    main()
