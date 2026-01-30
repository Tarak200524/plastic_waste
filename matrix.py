from ultralytics import YOLO
import torch

def main():
    print("CUDA available:", torch.cuda.is_available())

    model = YOLO("runs/detect/train10/weights/best.pt")

    metrics = model.val(
        data="data.yaml",
        device=0,        # GPU (CUDA)
        workers=4,       # use 2–4 for RTX 3050
        batch=8,         # reduce if VRAM issue
        save_json=True
    )

    print(metrics)

if __name__ == "__main__":
    main()
