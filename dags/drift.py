from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pymongo import MongoClient
from scipy.stats import ks_2samp

from src.config import IMAGES_DIR, MONGO_URI
from src.data import quality_features

FEATURES = ["blur", "glare"]
P_VALUE = 0.01      # below this, the two distributions are "different"
MIN_ROWS = 30       # too few predictions: don't judge
db = MongoClient(MONGO_URI).idverify


def reference(n=200):
    df = pd.read_csv(Path(IMAGES_DIR) / "labels.csv")
    paths = df[(df["split"] == "train") & (df["label"] == 0)]["path"]
    paths = paths.sample(n=min(n, len(paths)), random_state=0)
    return pd.DataFrame([quality_features(np.array(Image.open(p).convert("RGB"))) for p in paths])


def load_predictions(hours=24):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = db.predictions.find({"ts": {"$gte": since}}, {"_id": 0})
    return pd.DataFrame([{**r["quality"], "decision": r["decision"]} for r in rows])


def drift_report(cur):
    if len(cur) < MIN_ROWS:
        return {"n": len(cur), "drift": False, "note": "too few predictions"}
    ref = reference()
    p = {f: float(ks_2samp(ref[f], cur[f]).pvalue) for f in FEATURES}
    return {
        "n": len(cur),
        "p_values": p,
        "review_rate": float((cur["decision"] == "review").mean()),
        "retake_rate": float((cur["decision"] == "retake").mean()),
        "drift": any(v < P_VALUE for v in p.values()),
    }


def save_metrics(report):
    db.monitoring.insert_one({**report, "ts": datetime.now(timezone.utc)})