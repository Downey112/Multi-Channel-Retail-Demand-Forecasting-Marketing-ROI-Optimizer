"""(Re)build the schema and views, then load the simulated tables."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcr.db import ROOT, connect, run_sql_file  # noqa: E402

SIM = ROOT / "data" / "sim"
PROMO_DESC = {"Flash_Sale": "48-hour storewide markdown", "Bundle_Discount": "Multi-buy bundle pricing",
              "Loyalty_Cashback": "App cashback for loyalty members", "No_Promo": "Control: no promotion"}


def rows(df: pd.DataFrame) -> list[tuple]:
    return [tuple(v.item() if hasattr(v, "item") else v for v in r) for r in df.itertuples(index=False)]


def main() -> None:
    t = {n: pd.read_parquet(SIM / f"{n}.parquet") for n in
         ("stores", "weeks", "marketing_spend", "promo_assignments", "weekly_sales")}
    regions = {r: i + 1 for i, r in enumerate(sorted(t["stores"]["region"].unique()))}
    channels = {c: i + 1 for i, c in enumerate(t["marketing_spend"]["channel"].drop_duplicates())}
    with connect() as conn:
        run_sql_file(conn, "01_schema.sql")
        run_sql_file(conn, "02_views.sql")
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO regions VALUES (%s, %s)", [(i, r) for r, i in regions.items()])
            st = t["stores"].assign(region_id=t["stores"]["region"].map(regions))
            cur.executemany("INSERT INTO stores VALUES (%s, %s, %s, %s)",
                            rows(st[["store_id", "region_id", "store_format", "floor_area_sqft"]]))
            cur.executemany("INSERT INTO weeks VALUES (%s, %s, %s)", rows(t["weeks"]))
            cur.executemany("INSERT INTO channels VALUES (%s, %s)", [(i, c) for c, i in channels.items()])
            cur.executemany("INSERT INTO promo_types VALUES (%s, %s)", list(PROMO_DESC.items()))
            ms = t["marketing_spend"].assign(channel_id=t["marketing_spend"]["channel"].map(channels))
            cur.executemany("INSERT INTO marketing_spend VALUES (%s, %s, %s)",
                            rows(ms[["week_start", "channel_id", "spend_k"]]))
            cur.executemany("INSERT INTO promo_assignments VALUES (%s, %s, %s, %s)", rows(t["promo_assignments"]))
            cur.executemany("INSERT INTO weekly_sales VALUES (%s, %s, %s, %s)", rows(t["weekly_sales"]))
    for n, df in t.items():
        print(f"  loaded {n:<18} {len(df):>6,}")


if __name__ == "__main__":
    main()
