"""Module 1 (STA470 / experimental design): split-plot ANOVA of the randomised promo experiment."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcr import plotstyle as ps, report  # noqa: E402
from mcr.analysis import promo_experiment  # noqa: E402
from mcr.db import engine  # noqa: E402


def main() -> None:
    sw = pd.read_sql("SELECT * FROM vw_store_week ORDER BY store_id, week_start", engine())
    sw["revenue_rm"] = sw["revenue_rm"].astype(float)
    res = promo_experiment(sw)
    report.save("experiment", res)

    print(f"store-weeks {res['n_store_weeks']:,}")
    for r in res["anova"]:
        f = "" if r["f"] is None else f"F = {r['f']:8.2f}  p = {r['p']:.2e}  partial eta2 = {r['partial_eta_sq']:.3f}"
        print(f"  {r['source']:<34} df {r['df']:>6.0f}  {f}")
    print("promo contrasts (Tukey-Kramer, family 95%):")
    for t in res["tukey"]:
        print(f"  {t['comparison']:<38} {t['diff_rm']:>8,.0f}  [{t['ci_low']:>8,.0f}, {t['ci_high']:>8,.0f}]"
              f"  p_adj {t['p_adj']:.3g}")
    print(f"residual sd RM {res['residual_sd_rm']:,.0f}; mean within-store lag-1 autocorr "
          f"{res['mean_within_store_lag1_autocorr']:.3f}; Levene p {res['levene_p']:.3f}")

    se = pd.DataFrame(res["simple_effects"])
    regions = sorted(se["region"].unique())
    promos = ["Flash_Sale", "Bundle_Discount", "Loyalty_Cashback"]
    fig, ax = ps.plt.subplots(figsize=(8, 4.4))
    width = 0.26
    x = np.arange(len(regions))
    for i, p in enumerate(promos):
        d = se[se["promo"] == p].set_index("region").loc[regions]
        ax.bar(x + (i - 1) * width, d["lift_rm"], width, color=ps.SERIES[i], label=p.replace("_", " "))
        ax.errorbar(x + (i - 1) * width, d["lift_rm"],
                    yerr=[d["lift_rm"] - d["ci_low"], d["ci_high"] - d["lift_rm"]],
                    fmt="none", ecolor=ps.INK2, elinewidth=1, capsize=2)
    ax.set_xticks(x, [r.replace("_", " ") for r in regions])
    ax.set_ylabel("lift vs no promo (RM per store-week)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper right", fontsize=8)
    ax.set_title("Promo lift by region (95% CI), store and week fixed effects")
    ps.save(fig, "promo_lift_by_region.png")


if __name__ == "__main__":
    main()
