-- Optional bootstrap of BTC tables; tasks also create tables on the fly.
CREATE TABLE IF NOT EXISTS btc_5m (
  ts timestamptz PRIMARY KEY,
  open double precision, high double precision, low double precision,
  close double precision, volume double precision, source text DEFAULT 'tiingo'
);
CREATE TABLE IF NOT EXISTS btc_features_5m (
  ts timestamptz PRIMARY KEY,
  ret_5m double precision, vol_1h double precision, rsi_14 double precision,
  ema_12 double precision, ema_48 double precision, bb_upper double precision, bb_lower double precision
);
CREATE TABLE IF NOT EXISTS btc_forecast_1mo_path (
  ts_input timestamptz PRIMARY KEY,
  horizon_minutes int NOT NULL,
  step_minutes int NOT NULL,
  yhat_path double precision[],
  model_version text
);
CREATE TABLE IF NOT EXISTS btc_articles (
  id text PRIMARY KEY,
  published_at timestamptz,
  title text, url text, source text, body text
);
CREATE TABLE IF NOT EXISTS btc_sentiment (
  article_id text PRIMARY KEY REFERENCES btc_articles(id) ON DELETE CASCADE,
  compound double precision, pos double precision, neu double precision, neg double precision
);
CREATE TABLE IF NOT EXISTS btc_sentiment_rollups (
  ts_bucket timestamptz PRIMARY KEY,
  comp_30m double precision, comp_2h double precision, comp_1d double precision
);
CREATE TABLE IF NOT EXISTS model_registry (
  model_name text, version text, created_at timestamptz DEFAULT now(),
  binary bytea, meta jsonb, PRIMARY KEY (model_name, version)
);
CREATE TABLE IF NOT EXISTS rl_actions (
  ts timestamptz PRIMARY KEY,
  action text CHECK (action IN ('BUY','SELL','HOLD')),
  notional_usd double precision, price double precision, qty_btc double precision
);
CREATE TABLE IF NOT EXISTS portfolio (
  ts timestamptz PRIMARY KEY,
  btc_position double precision, usd_cash double precision, btc_nav double precision
);