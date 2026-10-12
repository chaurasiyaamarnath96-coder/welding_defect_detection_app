import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt

model = YOLO("C:\\Projects\\welding-defect-detection\\models\\best.pt")

results = model.predict(
    source="C:\\Projects\\welding-defect-detection\\dataset\\porosity-worm-hole-weld-defect-welding-process-flux-cored-arc-fcaw-204492534_jpeg_jpg.rf.c5201d1c888e3ffa1fded39930503d32.jpg",
    conf=0.25,
    save=True
)

