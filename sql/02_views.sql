-- Analysis views. scripts/03-05 read only these; mcr.analysis.store_week_frame / chain_week_frame
-- build the same frames in pandas, and tests/test_database.py checks that the two agree.

CREATE OR REPLACE VIEW vw_store_week AS
SELECT ws.store_id, r.name AS region, s.store_format, s.floor_area_sqft,
       ws.week_start, pa.promo_type, pa.block_id,
       ws.revenue_rm, ws.units_sold,
       w.festival, (w.festival <> '')::int AS festive, w.cpi_index
FROM weekly_sales ws
JOIN promo_assignments pa USING (store_id, week_start)
JOIN stores s  USING (store_id)
JOIN regions r USING (region_id)
JOIN weeks w   USING (week_start);

CREATE OR REPLACE VIEW vw_chain_week AS
WITH sales AS (
    SELECT week_start, SUM(revenue_rm) AS revenue_rm,
           COUNT(*) FILTER (WHERE promo_type = 'Flash_Sale')       AS n_flash_sale,
           COUNT(*) FILTER (WHERE promo_type = 'Bundle_Discount')  AS n_bundle_discount,
           COUNT(*) FILTER (WHERE promo_type = 'Loyalty_Cashback') AS n_loyalty_cashback
    FROM vw_store_week GROUP BY week_start
),
spend AS (
    SELECT ms.week_start,
           SUM(ms.spend_k) FILTER (WHERE c.name = 'Social_Video') AS spend_social_video_k,
           SUM(ms.spend_k) FILTER (WHERE c.name = 'Search_SEM')   AS spend_search_sem_k,
           SUM(ms.spend_k) FILTER (WHERE c.name = 'In_App_Push')  AS spend_in_app_push_k,
           SUM(ms.spend_k) FILTER (WHERE c.name = 'Out_Of_Home')  AS spend_out_of_home_k,
           SUM(ms.spend_k) AS total_spend_k
    FROM marketing_spend ms JOIN channels c USING (channel_id)
    GROUP BY ms.week_start
)
SELECT w.week_start, s.revenue_rm, (w.festival <> '')::int AS festive, w.cpi_index,
       s.n_flash_sale, s.n_bundle_discount, s.n_loyalty_cashback,
       sp.spend_social_video_k, sp.spend_search_sem_k, sp.spend_in_app_push_k, sp.spend_out_of_home_k,
       sp.total_spend_k,
       LAG(s.revenue_rm, 52) OVER (ORDER BY w.week_start)                    AS revenue_52w_ago_rm,
       AVG(s.revenue_rm) OVER (ORDER BY w.week_start ROWS 3 PRECEDING)       AS revenue_4w_avg_rm
FROM weeks w
JOIN sales s USING (week_start)
JOIN spend sp USING (week_start);

-- design check: every store sees each promo exactly once per 4-week block
CREATE OR REPLACE VIEW vw_rcbd_balance AS
SELECT store_id, block_id, COUNT(*) AS weeks, COUNT(DISTINCT promo_type) AS distinct_promos
FROM promo_assignments GROUP BY store_id, block_id;
