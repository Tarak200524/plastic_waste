from ultralytics import YOLO
import numpy as np
if __name__ == "__main__":
    model_path = r"C:\Users\tarun\OneDrive\Desktop\DATASET\runs\detect\train10\weights\best.pt"
    model = YOLO(model_path)
    metrics = model.val(workers=0)
    precision = float(metrics.box.p[0])
    recall = float(metrics.box.r[0])
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    print("\nOVERALL METRICS")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"mAP50     : {map50:.4f}")
    print(f"mAP50-95  : {map5095:.4f}")
