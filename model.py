from ultralytics import YOLO
import torch
print("GPU:", torch.cuda.get_device_name(0))
model = YOLO("yolo11m.pt")  
model.train(
    data="data.yaml",
    epochs=40,              
    imgsz=640,
    batch=4,            
    device=0,
    workers=0,               
    cache="ram",             

    optimizer="AdamW",
    lr0=0.0025,
    lrf=0.01,
    weight_decay=0.0003,
    momentum=0.937,


    mosaic=0.25,             
    mixup=0.10,
    copy_paste=0.10,
    hsv_s=0.45,
    hsv_v=0.45,
    fliplr=0.45,
    flipud=0.10,
    scale=0.55,
    erasing=0.40,            

   
    amp=False,


    freeze=None,

   
    patience=20,            
    pretrained=True,
)
