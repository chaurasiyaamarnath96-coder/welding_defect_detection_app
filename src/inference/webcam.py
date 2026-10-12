import cv2
from ultralytics import YOLO

model = YOLO("C:\\Projects\\welding-defect-detection\\models\\best.pt")

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow(
        "Welding Defect Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()