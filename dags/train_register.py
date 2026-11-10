from datetime import datetime
from airflow.sdk import dag, task
from src.data import split_dataset
from src.train import train_model, promote_if_better, export_onnx

ML = "/opt/airflow/venv-ml/bin/python"

@dag(schedule=None, start_date=datetime(2026, 1, 1), catchup=False)
def train_register():
    @task.external_python(python=ML)
    def prepare():
        from src.data import split_dataset, export_for_yolo
        split_dataset()
        export_for_yolo()

    @task.external_python(python=ML)
    def train():
        from src.train import train_model
        return str(train_model(epochs=10))       # MLflow run id

    @task.external_python(python=ML)
    def register(run_id):
        from src.train import promote_if_better
        return bool(promote_if_better(run_id))

    @task
    def reload_api(promoted):
        import requests
        if promoted:
            requests.post("http://api:8000/reload")

    run_id = train()
    prepare() >> run_id
    reload_api(register(run_id))


train_register()