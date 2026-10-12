from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=800,
    batch=16,
    patience = 30
)