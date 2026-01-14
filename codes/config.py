import os
import torch

# Paths
IMG_DIR = "/mnt/storage1/workspace/arobin/page_orientation/test_img"
SAVE_DIR = "/mnt/storage1/workspace/arobin/page_orientation/temp_output"
MODEL_PATH = "/mnt/storage1/workspace/arobin/page_orientation/models/deeplab_mobilenetv3_best.pth"
# YOLO_MODEL_PATH = "/mnt/storage1/workspace/arobin/page_orientation/models/best.pt"
CLASSIFIER_MODEL_PATH= "/mnt/storage1/workspace/arobin/page_orientation/models/rf_text_type_model.pkl"

# Model Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
IMG_SIZE = 512
SCALE_FACTOR = 0.95
IS_SAVE_INTERMEDIATE = True
YOLO_CONFIDENCE_THRESHOLD = 0.25 # Confidence threshold for YOLO model
YOLO_IOU_THRESHOLD = 0.45 # IoU threshold for YOLO model   
SR_SCALE = 2 
APPLY_SR = True 
HOG_CONFIG = None

os.makedirs(SAVE_DIR, exist_ok=True)