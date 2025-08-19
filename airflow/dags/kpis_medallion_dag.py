from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AIRFLOW_DATA = Path("/opt/airflow/data")

BRONZE = AIRFLOW_DATA / "bronze"
SILVER = AIRFLOW_DATA / "silver"
GOLD   = AIRFLOW_DATA / "gold"

SILVER_FILE = SILVER / "transactions_clean.parquet"
GOLD_PARQUET = GOLD / "kpis.parquet"

def _ensure_dirs():
    for p in (BRONZE, SILVER, GOLD):
        p.mkdir(parents=True, exist_ok=True)
        logger.info(f"Diretório garantido: {p}")

def _pick_latest_csv_in_bronze() -> Path:
    csvs = sorted(BRONZE.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("Nenhum CSV encontrado na camada bronze/")
    latest = csvs[-1]
    logger.info(f"Selecionado CSV mais recente na bronze: {latest}")
    return latest

def bronze_to_silver(**_):
    _ensure_dirs()
    src = _pick_latest_csv_in_bronze()
    logger.info(f"Lendo CSV cru: {src}")
    df = pd.read_csv(src)
    logger.info(f"Linhas lidas do bronze: {len(df)}")

    # Normalização de colunas
    cols = {c.lower().strip(): c for c in df.columns}
    rename_map = {}
    for cand in ("day", "date", "transaction_date"):
        if cand in cols:
            rename_map[cols[cand]] = "date"
            break
    if "amount_transacted" in cols: rename_map[cols["amount_transacted"]] = "amount_transacted"
    if "quantity_transactions" in cols: rename_map[cols["quantity_transactions"]] = "quantity_transactions"
    if "entity" in cols: rename_map[cols["entity"]] = "entity"

    df = df.rename(columns=rename_map)
    logger.info(f"Colunas após normalização: {df.columns.tolist()}")

    # Tipagem e validação
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["quantity_transactions"] = pd.to_numeric(df["quantity_transactions"], errors="coerce").fillna(0).astype(int)
    df["amount_transacted"] = pd.to_numeric(df["amount_transacted"], errors="coerce").fillna(0.0)

    # Coluna de auditoria: nome do arquivo de origem
    df["Arquivo_origem"] = src.name

    logger.info(f"Período: {df['date'].min()} até {df['date'].max()}")
    logger.info(f"Transações totais: {df['quantity_transactions'].sum()}, TPV total: {df['amount_transacted'].sum():,.2f}")

    df = df.drop_duplicates()
    df.to_parquet(SILVER_FILE, index=False)
    logger.info(f"Arquivo silver salvo em: {SILVER_FILE} ({len(df)} linhas)")

def silver_to_gold(**_):
    _ensure_dirs()
    if not SILVER_FILE.exists():
        raise FileNotFoundError("Arquivo silver inexistente.")

    logger.info(f"Lendo silver: {SILVER_FILE}")
    df = pd.read_parquet(SILVER_FILE)
    logger.info(f"Linhas no silver: {len(df)}")

    has_entity = "entity" in df.columns
    group_keys = ["date"] + (["entity"] if has_entity else [])

    daily = (
        df.groupby(group_keys + ["Arquivo_origem"])
          .agg(
              TPV=("amount_transacted","sum"),
              Transactions=("quantity_transactions","sum"),
          )
          .reset_index()
          .sort_values(group_keys)
    )
    daily["Average_Ticket"] = np.where(daily["Transactions"] > 0,
                                       daily["TPV"] / daily["Transactions"],
                                       np.nan)

    if has_entity:
        daily["TPV_MA7"] = (
            daily.sort_values(["entity","date"])
                 .groupby(["entity","Arquivo_origem"])["TPV"]
                 .transform(lambda s: s.rolling(7, min_periods=1).mean())
        )
    else:
        daily = daily.sort_values("date")
        daily["TPV_MA7"] = daily.groupby("Arquivo_origem")["TPV"].transform(lambda s: s.rolling(7, min_periods=1).mean())

    logger.info(f"Linhas agregadas na gold: {len(daily)}")
    logger.info(f"Período gold: {daily['date'].min()} até {daily['date'].max()}")
    logger.info(f"TPV total gold: {daily['TPV'].sum():,.2f}")

    daily.to_parquet(GOLD_PARQUET, index=False)
    logger.info(f"Arquivos salvos: {GOLD_PARQUET}")

with DAG(
    dag_id="medallion_kpis",
    description="Processamento de dados Bronze→Silver→Gold",
    start_date=datetime(2025, 8, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args={"owner": "lucas", "retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["medallion","kpis"],
) as dag:

    t_silver = PythonOperator(task_id="bronze_to_silver", python_callable=bronze_to_silver)
    t_gold   = PythonOperator(task_id="silver_to_gold",   python_callable=silver_to_gold)

    t_silver >> t_gold