from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
from plugins.jobs.collect_tiingo import collect_prices_5m, collect_news
from plugins.jobs.features import update_features_5m
from plugins.jobs.sentiment import score_new_articles, rollup_sentiment
from plugins.jobs.forecast import run_inference_path
from plugins.jobs.rl import rl_decide_and_execute

default_args = dict(owner='you', retries=2, retry_delay=timedelta(minutes=2))

with DAG(
    'btc_ingest_5m',
    start_date=datetime(2025,1,1, tzinfo=timezone.utc),
    schedule_interval='*/5 * * * *',
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=['btc','tiingo','rl','lstm'],
) as dag:

    t_prices = PythonOperator(task_id='collect_prices_5m', python_callable=collect_prices_5m)
    t_news   = PythonOperator(task_id='collect_news', python_callable=collect_news)
    t_feat   = PythonOperator(task_id='update_features_5m', python_callable=update_features_5m)

    def _sent():
        score_new_articles(); rollup_sentiment()

    t_sent   = PythonOperator(task_id='score_and_rollup_sentiment', python_callable=_sent)
    t_infer  = PythonOperator(task_id='forecast_path_1mo', python_callable=run_inference_path)
    t_rl     = PythonOperator(task_id='rl_decide_execute', python_callable=rl_decide_and_execute)

    [t_prices, t_news] >> t_feat >> t_sent >> t_infer >> t_rl