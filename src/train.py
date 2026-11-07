import os

import mlflow
from mlflow import MlflowClient
from ultralytics import YOLO, settings

from src.config import MLFLOW_URI

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
MODEL_NAME = "doc-verifier"
METRIC = "metrics/accuracy_top1"   # confirm the exact name in the UI


def train_model(imgsz=256, epochs=50):
    settings.update({"mlflow": True})
    model = YOLO("yolo11n-cls.pt")
    model.train(data="data/cls", epochs=epochs, imgsz=imgsz, batch=32, patience=15,
                project=MODEL_NAME, name=f"cls-{imgsz}")
    return latest_run_id()


def latest_run_id():
    exp = mlflow.get_experiment_by_name(MODEL_NAME)
    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    return runs.loc[0, "run_id"]


def promote_if_better(run_id):
    client = MlflowClient()
    new = client.get_run(run_id).data.metrics[METRIC]
    try:
        prod = client.get_model_version_by_alias(MODEL_NAME, "production")
        old = client.get_run(prod.run_id).data.metrics[METRIC]
    except Exception:          # no production model yet
        old = -1
    if new > old:
        mv = mlflow.register_model(f"runs:/{run_id}/weights/best.pt", MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "production", mv.version)
    return new > old