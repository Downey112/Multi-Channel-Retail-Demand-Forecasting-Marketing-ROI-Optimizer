-- DBeaver workbook: connection localhost:5434, database retail_roi, user postgres, password roi.
-- Run statements one at a time (Ctrl+Enter).

-- 1. Design check: each store sees each promo exactly once per 4-week block (expect 0 rows)
SELECT * FROM vw_rcbd_balance WHERE weeks <> 4 OR distinct_promos <> 4;

-- 2. Raw promo means by region (the model-based estimates in docs/results/experiment.json
--    also remove store and week effects)
SELECT region, promo_type, COUNT(*) AS store_weeks, ROUND(AVG(revenue_rm)) AS avg_revenue_rm
FROM vw_store_week
GROUP BY region, promo_type
ORDER BY region, avg_revenue_rm DESC;

-- 3. Lift vs No_Promo within store and week (paired comparison, no model needed)
WITH base AS (
    SELECT store_id, block_id, AVG(revenue_rm) FILTER (WHERE promo_type = 'No_Promo') AS no_promo
    FROM vw_store_week GROUP BY store_id, block_id
)
SELECT v.promo_type, ROUND(AVG(v.revenue_rm - b.no_promo)) AS avg_within_block_lift_rm
FROM vw_store_week v JOIN base b USING (store_id, block_id)
WHERE v.promo_type <> 'No_Promo'
GROUP BY v.promo_type ORDER BY 2 DESC;

-- 4. Chain week with media mix, 52-week-ago revenue and a 4-week moving average
SELECT week_start, revenue_rm, revenue_52w_ago_rm,
       ROUND(100.0 * (revenue_rm / NULLIF(revenue_52w_ago_rm, 0) - 1), 1) AS yoy_pct,
       ROUND(revenue_4w_avg_rm) AS avg_4w, total_spend_k, festive
FROM vw_chain_week ORDER BY week_start DESC LIMIT 20;

-- 5. Festive weeks: revenue and media spend vs normal weeks
SELECT w.festival = '' AS normal_week, COUNT(*) AS weeks,
       ROUND(AVG(c.revenue_rm)) AS avg_revenue_rm, ROUND(AVG(c.total_spend_k), 1) AS avg_spend_k
FROM vw_chain_week c JOIN weeks w USING (week_start)
GROUP BY 1;

-- 6. Real CPI (OpenDOSM) on the weekly grid: month-on-month change
SELECT DISTINCT date_trunc('month', week_start)::date AS month, cpi_index
FROM weeks ORDER BY month;

-- 7. Latest forecast run: both media plans side by side
SELECT f.week_start,
       MAX(f.point_rm) FILTER (WHERE scenario = 'current_plan')          AS current_plan_rm,
       MAX(f.point_rm) FILTER (WHERE scenario = 'profit_optimal_budget') AS optimal_budget_rm,
       MAX(f.planned_spend_k) FILTER (WHERE scenario = 'current_plan')          AS current_spend_k,
       MAX(f.planned_spend_k) FILTER (WHERE scenario = 'profit_optimal_budget') AS optimal_spend_k
FROM forecasts f
WHERE f.run_id = (SELECT MAX(run_id) FROM model_runs)
GROUP BY f.week_start ORDER BY f.week_start;
