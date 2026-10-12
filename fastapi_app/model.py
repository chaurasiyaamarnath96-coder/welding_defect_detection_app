# app/model.py

from ultralytics import YOLO

MODEL_PATH = "C:\\Projects\\welding-defect-detection\\models\\best.pt"

model = YOLO(MODEL_PATH)