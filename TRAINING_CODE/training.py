from torch import device
from ultralytics import YOLO 

model = YOLO('yolov8s-seg.pt')

result = model.train(
data = "/mnt/storage1/workspace/arobin/page_orientation/data/splited_data_with_4_class/data.yml",
epochs =100,
imgsz = 640,
batch =8 ,
patience=15,
verbose=True, 
project = "/mnt/storage1/workspace/arobin/page_orientation/seg_with_4class/runs",
name="train_seg_4class_v1",
device=0)