🏭 Welding Defect Detection Using YOLOv11n
An AI-powered computer vision system for automatic welding quality inspection using YOLOv11n, FastAPI, and Streamlit. The application detects welding defects from images and live webcam feeds, classifies weld quality, and provides real-time visualization and reporting.

📌 Overview
Welding quality inspection is a critical process in manufacturing industries where manual inspection can be time-consuming, inconsistent, and prone to human error. This project leverages deep learning-based object detection to identify welding defects and classify weld quality automatically.

The system is trained on a custom welding defect dataset and deployed through a user-friendly dashboard that supports both image uploads and live camera inspection.

🚀 Features
✅ Real-time Welding Defect Detection

✅ Powered by YOLOv11n Object Detection Model

✅ Upload and Analyze Welding Images

✅ Live Webcam Inspection

✅ Bounding Box Visualization

✅ Confidence Score Display

✅ Defect Summary Dashboard

✅ JSON Report Generation

✅ Download Inspection Reports

✅ FastAPI Backend Integration

✅ Interactive Streamlit Frontend

🧠 Defect Classes
The model is trained to detect the following classes:

Good Weld
Bad Weld
Defect
⚙️ Tech Stack
Machine Learning
YOLOv11n
PyTorch
OpenCV
NumPy
Backend
FastAPI
Uvicorn
Frontend
Streamlit
Plotly
Deployment
Docker (Optional)
📂 Project Structure
welding-defect-detection/
│
├── app/
│   ├── main.py
│   ├── model.py
│
├── models/
│   └── best.pt
│
├── uploads/
│
├── dataset/
│
├── streamlit_app.py
│
├── requirements.txt
│
└── README.md
🔄 System Workflow
User Uploads Image / Webcam Frame
                │
                ▼
          Streamlit UI
                │
                ▼
          FastAPI Backend
                │
                ▼
       YOLOv11n Inference
                │
                ▼
      Defect Classification
                │
                ▼
       Visualization & Report
                │
                ▼
             User
🏗️ Model Training
The YOLOv11n model was trained on a welding defect dataset obtained from Kaggle.

Training Configuration
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
🎯 Quality Assessment Logic
The system determines weld quality based on detected classes and confidence scores.

GOOD WELD
No defects detected
Only "Good Weld" predictions detected
BAD WELD
Defect detected above confidence threshold
Bad Weld class detected above confidence threshold
CONFIDENCE_THRESHOLD = 0.50

if class_name in ["Defect", "Bad Weld"] \
    and confidence >= CONFIDENCE_THRESHOLD:
    total_defects += 1
📊 Dashboard Features
The Streamlit dashboard provides:

Original Image Preview
Predicted Image Preview
Detection Results Table
Confidence Score Visualization
Quality Status Indicator
Downloadable JSON Report
Live Webcam Detection
📷 Sample Output
Inspection Status: BAD WELD ❌

Detected Classes:
- Defect (94.5%)

Total Defects: 1
Highest Confidence: 94.5%
🛠️ Installation
Clone Repository
git clone https://github.com/yourusername/welding-defect-detection.git

cd welding-defect-detection
Create Virtual Environment
python -m venv myenv
Activate Environment
Windows
myenv\Scripts\activate
Linux/Mac
source myenv/bin/activate
Install Dependencies
pip install -r requirements.txt
▶️ Run FastAPI Backend
uvicorn app.main:app --reload
Backend URL:

http://127.0.0.1:8000
Swagger Documentation:

http://127.0.0.1:8000/docs
▶️ Run Streamlit Dashboard
streamlit run streamlit_app.py
Application URL:

https://weldingdefectdetectionapp-a8bnbvcc8tapuqcjcd5ccb.streamlit.app/
📈 Future Improvements
Defect Severity Analysis
Multi-Class Welding Defect Detection
ONNX / TensorRT Optimization
Industrial Camera Integration
Cloud Deployment
Defect Tracking in Video Streams
Automated Inspection Reports
🎓 Learning Outcomes
This project demonstrates:

Object Detection with YOLOv11n
Transfer Learning
Computer Vision Applications
FastAPI Backend Development
Streamlit Dashboard Development
Real-Time Inference
End-to-End Machine Learning Deployment
👨‍💻 Author
Amar Nath

B.Tech Computer Science Engineering

Machine Learning & Computer Vision Enthusiast