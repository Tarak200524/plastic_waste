import cv2
import torch
from ultralytics import YOLO

def predict(image_path):
    print("GPU:", torch.cuda.get_device_name(0))
    model = YOLO("runs/detect/train10/weights/best.pt").to("cuda")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found")

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8)).apply(l)
    img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    img = cv2.addWeighted(img, 1.6, cv2.GaussianBlur(img, (5, 5), 0), -0.6, 0)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    result = model.predict(
        img,
        conf=0.25,
        iou=0.5,
        imgsz=640,
        device=0,
        half=True,
        verbose=False
    )[0]

    annotated = result.plot()
    resized = cv2.resize(annotated, (412, 600))

    cv2.imshow("Prediction", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict(r"C:\Users\tarun\OneDrive\Desktop\DATASET\images\train\ps31.webp")
