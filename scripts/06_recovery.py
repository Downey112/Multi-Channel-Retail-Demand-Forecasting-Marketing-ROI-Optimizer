"""Parameter recovery: compare every estimate with the simulator's true value.

This is the only analysis step allowed to read the truth (data/sim/true_params.json). It runs after
modules 1-3 have written their results, so it cannot influence them.

  python scripts/06_recovery.py             # recovery table for this dataset
  python scripts/06_recovery.py --reps 100  # plus a Monte Carlo study: re-simulate with new seeds,
                                            # re-run the estimators, report bias and CI coverage
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcr import report  # noqa: E402
from mcr.analysis import (CHANNELS, bootstrap_response, fit_response, promo_experiment,  # noqa: E402
                          response_params, roi_summary, chain_week_frame, store_week_frame)
from mcr.calendar import monthly_cpi  # noqa: E402
from mcr.db import ROOT  # noqa: E402

SIM = ROOT / "data" / "sim"
PROMOS = ["Flash_Sale", "Bundle_Discount", "Loyalty_Cashback"]
REGIONS = ["Central", "East_Coast", "Northern", "Southern"]


def truth_values(T: dict, mix: dict) -> dict:
    """True values in the units each estimator reports."""
    inter = T["interaction_rm"]
    v = {f"lift:{p}": T["promo_rm"][p] + np.mean([inter.get(f"{r}:{p}", 0.0) for r in REGIONS]) for p in PROMOS}
    v.update({f"cell:{r}:{p}": T["promo_rm"][p] + inter.get(f"{r}:{p}", 0.0) for r in REGIONS for p in PROMOS})
    n = len(REGIONS) * 3
    w = T["channel_weight_rel"]
    v.update({"adstock_lambda": T["adstock_lambda"], "c": T["adbudg_c"], "half_k": T["adbudg_half_k"],
              "w_search_sem": w["Search_SEM"], "w_in_app_push": w["In_App_Push"], "w_out_of_home": w["Out_Of_Home"],
              "ceiling_rm": T["adbudg_max_rm"], "festive_rm": n * T["festive_rm"],
              "cpi_rm_per_point": n * T["cpi_rm_per_point"]})
    theta = {k: v[k] for k in ("adstock_lambda", "c", "half_k", "w_search_sem", "w_in_app_push", "w_out_of_home")}
    roi = roi_summary(theta, T["adbudg_max_rm"], mix, report.load("response")["roi"]["margin_assumed"])
    v["inflection_spend_k"], v["optimal_spend_k"] = roi["inflection_spend_k"], roi["optimal_spend_k"]
    return v


def estimates(exp: dict, resp: dict) -> dict:
    """{name: (estimate, ci_low, ci_high)} from the saved module results."""
    e = {f"lift:{d['promo']}": (d["lift_rm"], d["ci_low"], d["ci_high"]) for d in exp["promo_lift"]}
    e.update({f"cell:{d['region']}:{d['promo']}": (d["lift_rm"], d["ci_low"], d["ci_high"])
              for d in exp["simple_effects"]})
    est = {**resp["adbudg"]["theta"], "ceiling_rm": resp["adbudg"]["beta"]["ceiling_rm"],
           "festive_rm": resp["adbudg"]["beta"]["festive"], "cpi_rm_per_point": resp["adbudg"]["beta"]["cpi_dev"]}
    e.update({k: (v, *resp["adbudg_ci95"][k]) for k, v in est.items()})
    for k in ("inflection_spend_k", "optimal_spend_k"):
        e[k] = (resp["roi"][k], *resp["roi_ci95"][k])
    return e


def recovery_table() -> pd.DataFrame:
    T = json.loads((SIM / "true_params.json").read_text(encoding="utf-8"))
    resp = report.load("response")
    tv, est = truth_values(T, resp["channel_mix_last52"]), estimates(report.load("experiment"), resp)
    rows = [{"parameter": k, "true": tv[k], "estimate": e, "ci_low": lo, "ci_high": hi,
             "covered": bool(lo <= tv[k] <= hi)} for k, (e, lo, hi) in est.items()]
    return pd.DataFrame(rows)


def monte_carlo(reps: int, n_boot: int) -> dict:
    T = json.loads((SIM / "true_params.json").read_text(encoding="utf-8"))
    from mcr.simulate import simulate          # Monte Carlo needs fresh draws; still outside mcr.analysis
    cpi = monthly_cpi(ROOT / "data" / "raw" / "cpi_2d.parquet")
    mix = report.load("response")["channel_mix_last52"]
    tv = truth_values(T, mix)
    margin = report.load("response")["roi"]["margin_assumed"]
    rows, t0 = [], time.time()
    for r in range(reps):
        t = simulate(cpi, seed=10_000 + r)
        exp = promo_experiment(store_week_frame(t))
        ch = chain_week_frame(t)
        fit = fit_response(ch)
        boot = bootstrap_response(ch, fit, n_boot=n_boot, seed=r)
        row = {"rep": r}
        for d in exp["promo_lift"]:
            k = f"lift:{d['promo']}"
            row[k], row[k + ":cov"] = d["lift_rm"], d["ci_low"] <= tv[k] <= d["ci_high"]
        cells = [f"cell:{d['region']}:{d['promo']}" for d in exp["simple_effects"]]
        row["cells:cov_share"] = np.mean([d["ci_low"] <= tv[c] <= d["ci_high"]
                                         for c, d in zip(cells, exp["simple_effects"])])
        cats = {"Flash_Sale": tv["lift:Flash_Sale"], "Bundle_Discount": tv["lift:Bundle_Discount"],
                "Loyalty_Cashback": tv["lift:Loyalty_Cashback"], "No_Promo": 0.0}
        row["tukey:family_cov"] = all(
            d["ci_low"] <= cats[d["comparison"].split(" vs ")[0]] - cats[d["comparison"].split(" vs ")[1]] <= d["ci_high"]
            for d in exp["tukey"])
        for k, v in response_params(fit).items():
            lo, hi = boot[k].quantile([0.025, 0.975])
            row[k], row[k + ":cov"] = v, lo <= tv[k] <= hi
        roi = roi_summary(fit["theta"], fit["beta"]["ceiling_rm"], mix, margin)
        opt_b = [roi_summary({k: b[k] for k in fit["theta"]}, b["ceiling_rm"], mix, margin)["optimal_spend_k"]
                 for _, b in boot.iterrows()]
        lo, hi = np.nanquantile(opt_b, [0.025, 0.975])
        row["optimal_spend_k"], row["optimal_spend_k:cov"] = roi["optimal_spend_k"], lo <= tv["optimal_spend_k"] <= hi
        rows.append(row)
        print(f"  rep {r + 1}/{reps}  ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    summary = []
    for k in [c for c in df.columns if c.endswith(":cov") and not c.startswith("cells")]:
        name = k[:-4]
        est = df[name].astype(float)
        summary.append({"parameter": name, "true": tv[name], "mean_estimate": float(est.mean()),
                        "median_estimate": float(est.median()), "bias_pct": float((est.mean() / tv[name] - 1) * 100),
                        "coverage_95": float(df[k].mean())})
    return {"reps": reps, "n_boot": n_boot, "parameters": summary,
            "simple_effect_ci_coverage": float(df["cells:cov_share"].mean()),
            "tukey_family_coverage": float(df["tukey:family_cov"].mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--boot", type=int, default=99)
    a = ap.parse_args()
    tab = recovery_table()
    pd.set_option("display.width", 140)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"covered by their 95% CI: {tab['covered'].sum()} of {len(tab)}")
    out = {"table": tab.to_dict("records")}
    if a.reps:
        out["monte_carlo"] = monte_carlo(a.reps, a.boot)
        print(pd.DataFrame(out["monte_carlo"]["parameters"]).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
        print(f"simple-effect CI coverage {out['monte_carlo']['simple_effect_ci_coverage']:.3f}; "
              f"Tukey family coverage {out['monte_carlo']['tukey_family_coverage']:.3f}")
    elif (report.RESULTS / "recovery.json").exists():
        prev = report.load("recovery")
        if "monte_carlo" in prev:
            out["monte_carlo"] = prev["monte_carlo"]          # keep the last (slow) Monte Carlo run
    report.save("recovery", out)


if __name__ == "__main__":
    main()
