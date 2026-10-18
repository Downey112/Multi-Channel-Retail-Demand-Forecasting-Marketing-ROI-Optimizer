"""Fetch the real inputs (OpenDOSM CPI) and simulate the chain. Writes data/sim/*.parquet.

The true parameters go to data/sim/true_params.json, which only scripts/06_recovery.py reads.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcr.calendar import CPI_URL, monthly_cpi  # noqa: E402
from mcr.db import ROOT  # noqa: E402
from mcr.simulate import simulate, truth_for_report  # noqa: E402

RAW, SIM = ROOT / "data" / "raw", ROOT / "data" / "sim"


def fetch(url: str, dest: Path, tries: int = 6) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):      # storage.dosm.gov.my sometimes resets connections
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"downloaded {url}")
            return
        except OSError as e:
            print(f"  attempt {i + 1} failed: {e}")
            time.sleep(2 * (i + 1))
    raise SystemExit(f"could not download {url}")


def main() -> None:
    fetch(CPI_URL, RAW / "cpi_2d.parquet")
    tables = simulate(monthly_cpi(RAW / "cpi_2d.parquet"))
    SIM.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(SIM / f"{name}.parquet", index=False)
        print(f"  {name:<18} {len(df):>6,} rows")
    (SIM / "true_params.json").write_text(json.dumps(truth_for_report(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
