from datetime import datetime
from airflow.sdk import dag, task
from src.data import split_dataset
from src.train import train_model, promote_if_better, export_onnx

@dag(schedule=None, start_date=datetime(2026, 1, 1), catchup=False)
def train_register():
    @task
    def prepare(): split_dataset()
    @task
    def train(): return train_model()      
    @task
    def register(run_id): promote_if_better(run_id)
    @task
    def export(run_id): export_onnx(run_id)

    run_id = train()
    prepare() >> run_id
    register(run_id)
    export(run_id)

train_register()