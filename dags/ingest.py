from datetime import datetime
from airflow.sdk import dag, task


@dag(schedule=None, start_date=datetime(2026, 1, 1), catchup=False)
def ingest():
    @task
    def build():
        from src.data import build_dataset
        build_dataset(n=300)

    @task
    def validate():
        from src.data import validate_dataset
        validate_dataset()

    build() >> validate()


ingest()