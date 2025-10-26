from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
from plugins.jobs.train_lstm import train_lstm_seq2seq_3y
from plugins.jobs.train_rl import train_dqn_daily

default_args = dict(owner='you', retries=1, retry_delay=timedelta(minutes=10))

with DAG(
    'btc_train_daily',
    start_date=datetime(2025,1,1, tzinfo=timezone.utc),
    schedule_interval='0 3 * * *',
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=['train','btc'],
) as dag:
    t_price_model = PythonOperator(task_id='train_lstm_seq2seq_3year', python_callable=train_lstm_seq2seq_3y)
    t_rl_model    = PythonOperator(task_id='train_dqn_from_buffer', python_callable=train_dqn_daily)
    t_price_model >> t_rl_model