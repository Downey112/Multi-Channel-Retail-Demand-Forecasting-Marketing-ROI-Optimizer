-- Multi-channel retail demand & marketing ROI: physical schema (PostgreSQL 14+).
-- Store sales, promotions and media spend are SIMULATED (src/mcr/simulate.py); the festive calendar
-- and CPI are real (Malaysian public holidays, OpenDOSM headline CPI).
DROP TABLE IF EXISTS forecasts, model_runs, weekly_sales, promo_assignments, marketing_spend,
                     promo_types, channels, weeks, stores, regions CASCADE;

CREATE TABLE regions (
    region_id  smallint PRIMARY KEY,
    name       text     NOT NULL UNIQUE
);

CREATE TABLE stores (
    store_id         text     PRIMARY KEY,
    region_id        smallint NOT NULL REFERENCES regions,
    store_format     text     NOT NULL CHECK (store_format IN ('Flagship_Mall','Hypermarket','Suburban_Hub')),
    floor_area_sqft  integer  NOT NULL CHECK (floor_area_sqft > 0)
);

CREATE TABLE weeks (
    week_start  date          PRIMARY KEY CHECK (EXTRACT(ISODOW FROM week_start) = 1),   -- Mondays
    festival    text          NOT NULL DEFAULT '',          -- REAL: festival week or the week before it
    cpi_index   numeric(6,1)  NOT NULL                       -- REAL: OpenDOSM headline CPI of that month
);
COMMENT ON COLUMN weeks.festival IS 'Real Malaysian festival (CNY, Aidilfitri, Deepavali, Christmas) whose date falls in this week or the next.';

CREATE TABLE channels (
    channel_id  smallint PRIMARY KEY,
    name        text     NOT NULL UNIQUE
);

CREATE TABLE promo_types (
    promo_type   text PRIMARY KEY,
    description  text NOT NULL
);

CREATE TABLE marketing_spend (                 -- SIMULATED, chain level
    week_start  date          NOT NULL REFERENCES weeks,
    channel_id  smallint      NOT NULL REFERENCES channels,
    spend_k     numeric(9,3)  NOT NULL CHECK (spend_k >= 0),     -- RM '000
    PRIMARY KEY (week_start, channel_id)
);

CREATE TABLE promo_assignments (               -- SIMULATED randomised complete block design
    store_id    text      NOT NULL REFERENCES stores,
    week_start  date      NOT NULL REFERENCES weeks,
    promo_type  text      NOT NULL REFERENCES promo_types,
    block_id    smallint  NOT NULL,                              -- 4-week block; each promo once per block
    PRIMARY KEY (store_id, week_start)
);

CREATE TABLE weekly_sales (                    -- SIMULATED
    store_id    text           NOT NULL,
    week_start  date           NOT NULL,
    revenue_rm  numeric(12,2)  NOT NULL CHECK (revenue_rm > 0),
    units_sold  integer        NOT NULL CHECK (units_sold > 0),
    PRIMARY KEY (store_id, week_start),
    FOREIGN KEY (store_id, week_start) REFERENCES promo_assignments
);

CREATE TABLE model_runs (
    run_id           serial       PRIMARY KEY,
    model            text         NOT NULL,
    spec             text         NOT NULL,
    trained_through  date         NOT NULL REFERENCES weeks,
    created_at       timestamptz  NOT NULL DEFAULT now()
);

CREATE TABLE forecasts (
    run_id       integer        NOT NULL REFERENCES model_runs ON DELETE CASCADE,
    scenario     text           NOT NULL,
    week_start   date           NOT NULL CHECK (EXTRACT(ISODOW FROM week_start) = 1),
    planned_spend_k numeric(9,3) NOT NULL,
    point_rm     numeric(14,2)  NOT NULL,
    lo95_rm      numeric(14,2)  NOT NULL,
    hi95_rm      numeric(14,2)  NOT NULL,
    PRIMARY KEY (run_id, scenario, week_start),
    CHECK (lo95_rm <= point_rm AND point_rm <= hi95_rm)
);

CREATE INDEX ix_sales_week ON weekly_sales (week_start);
CREATE INDEX ix_promo_week ON promo_assignments (week_start, promo_type);
