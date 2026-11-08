import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pymongo import MongoClient
from ultralytics import YOLO

from src.config import IMAGES_DIR, MLFLOW_URI, MONGO_URI
from src.data import quality_features, quality_ok

MODEL_NAME = "doc-verifier"
REJECT, REVIEW = 0.8, 0.4        # spoof_score thresholds, tune later

mlflow.set_tracking_uri(MLFLOW_URI)
predictions = MongoClient(MONGO_URI).idverify.predictions
os.makedirs(f"{IMAGES_DIR}/predictions", exist_ok=True)
state = {}


def load_model():
    mv = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "production")
    path = mlflow.artifacts.download_artifacts(f"runs:/{mv.run_id}/weights/best.pt")
    state["model"] = YOLO(path)
    state["version"] = mv.version
    state["fake_idx"] = [i for i, n in state["model"].names.items() if n == "fake"][0]


@asynccontextmanager
async def lifespan(app):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


def decide(q, spoof):
    if not quality_ok(q):
        return "retake"          # bad photo, not fraud
    if spoof >= REJECT:
        return "reject"
    if spoof >= REVIEW:
        return "review"          # goes to the human queue
    return "accept"


@app.post("/verify")
def verify(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    img.thumbnail((1024, 1024))                      # same as in training
    q = quality_features(np.array(img))
    probs = state["model"](img, verbose=False)[0].probs.data
    spoof = float(probs[state["fake_idx"]])

    pid = uuid.uuid4().hex
    path = f"{IMAGES_DIR}/predictions/{pid}.jpg"
    img.save(path, quality=90)

    result = {"id": pid, "decision": decide(q, spoof), "spoof_score": spoof,
              "quality": q, "model_version": state["version"]}
    predictions.insert_one({**result, "ts": datetime.now(timezone.utc), "image": path})
    return result


@app.post("/reload")
def reload():
    load_model()
    return {"model_version": state["version"]}


@app.get("/health")
def health():
    return {"status": "ok", "model_version": state.get("version")}