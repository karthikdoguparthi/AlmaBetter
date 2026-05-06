from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Add current directory to path to import train_models
sys.path.append(os.path.dirname(__file__))
# try:
#     from train_models import train_models
# except ImportError:
#     def train_models():
#         print("Training logic not found in path")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def data_ingestion():
    print("Ingesting new travel data from source...")
    # Logic to fetch new data would go here

def data_preprocessing():
    print("Preprocessing ingested data...")
    # Logic to clean and prepare data would go here

def model_retraining():
    print("Starting model retraining...")
    # Call the train_models function
    # train_models()

with DAG(
    'travel_ml_pipeline',
    default_args=default_args,
    description='A DAG for travel data regression and classification pipeline',
    schedule_interval=timedelta(days=1),
) as dag:

    t1 = PythonOperator(
        task_id='ingest_data',
        python_callable=data_ingestion,
    )

    t2 = PythonOperator(
        task_id='preprocess_data',
        python_callable=data_preprocessing,
    )

    t3 = PythonOperator(
        task_id='retrain_models',
        python_callable=model_retraining,
    )

    t1 >> t2 >> t3
