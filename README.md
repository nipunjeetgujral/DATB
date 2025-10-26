# Tiingo BTC — Airflow + Postgres micro-stack

Quick start
-----------
1) Copy env and set your token:
   cp .env.example .env  # then edit .env to set TIINGO_TOKEN

2) Launch:
   docker compose up --build

3) Open Airflow: http://localhost:8080 (admin/admin)
   Unpause DAGs: btc_ingest_5m and btc_train_daily