import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")
IMAGES_DIR = os.getenv("IMAGES_DIR", "data/images")
API_URL = os.getenv("API_URL", "http://localhost:8000")
CLS_DIR = os.getenv("CLS_DIR", "data/cls")