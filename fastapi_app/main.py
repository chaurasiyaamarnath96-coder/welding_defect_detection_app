import os

from fastapi import FastAPI, UploadFile, File
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from model import model
import uuid
import cv2

app = FastAPI(
    title="Welding Defect Detection API"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
os.makedirs("C:\\Projects\\welding-defect-detection\\uploads", exist_ok=True)

@app.get("/")
def home():
    return {
        "message": "YOLOv11 Welding Defect API Running"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File()
):

    image = Image.open(file.file)

    results = model(image)
    detections = []
    total_defects = 0
    max_confidence = 0.0

    for box in results[0].boxes:
        cls_id = int(box.cls[0])

        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        max_confidence = max(
            max_confidence,confidence
        )
        
        bbox = box.xyxy[0].tolist()

        
        total_defects += 1

            
        detections.append({

        "class_name": class_name,

        "confidence": max_confidence,

        "bbox": bbox
        })
    annotated_image = results[0].plot()
    filename = f"{uuid.uuid4()}.jpg"
    save_path = os.path.join("C:\\Projects\\welding-defect-detection\\uploads", filename)
    cv2.imwrite(save_path, annotated_image)

    return {
        "detections": detections,
        "total_defects": total_defects,
        "max_confidence": round(max_confidence*100, 2),
        "prediction_image": save_path
    }
                