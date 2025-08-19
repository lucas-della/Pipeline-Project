from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from pathlib import Path

def _make_kpis():
    Path("/opt/airflow/data").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "date": pd.date_range("2025-07-01", periods=30, freq="D"),
        "TPV": [1000 + i*15 for i in range(30)],
        "Transactions": [10 + (i % 5) for i in range(30)]
    })
    df["Average_Ticket"] = df["TPV"] / df["Transactions"]
    df.to_csv("/opt/airflow/data/kpis_gold.csv", index=False)

with DAG(
    dag_id="make_kpis_csv",
    start_date=datetime(2024,1,1),
    schedule=None,
    catchup=False,
    tags=["check"],
):
    t = PythonOperator(task_id="make", python_callable=_make_kpis)