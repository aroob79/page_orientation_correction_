import os
import cv2
import torch
import numpy as np
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

# ---------------- CONFIG ----------------
IMG_DIR = "/mnt/storage1/workspace/arobin/bangla_printed_ocr/bangla_ocr/page_orientation/test_img/photo_6316441238265072542_y.jpg"  # folder of images or single image
SAVE_DIR = "/mnt/storage1/workspace/arobin/bangla_printed_ocr/bangla_ocr/page_orientation/using_deeplabv3/predicted_masks"
MODEL_PATH = "/mnt/storage1/workspace/arobin/bangla_printed_ocr/bangla_ocr/page_orientation/using_deeplabv3/models/deeplab_mobilenetv3_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
IMG_SIZE = 512  # resize images to 512x512

os.makedirs(SAVE_DIR, exist_ok=True)

# define model
model = deeplabv3_mobilenet_v3_large(weights=None)
model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, 1)
model = model.to(DEVICE)

# load state_dict safely (ignore aux_classifier keys)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = {k: v for k, v in state_dict.items() if "aux_classifier" not in k}  # remove aux
model.load_state_dict(state_dict, strict=False)

model.eval()
# ---------------- INFERENCE FUNCTION ----------------
def predict_mask(img_path):
    img_name = os.path.basename(img_path)
    
    # Read and preprocess
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_norm = (img_resized - mean) / std
    tensor = torch.tensor(img_norm).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

    # Predict
    with torch.no_grad():
        out = model(tensor)["out"]
        mask = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Resize back to original size
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Save mask
    save_path = os.path.join(SAVE_DIR, img_name.replace(".jpg", ".png"))
    cv2.imwrite(save_path, mask*255)  # optional: multiply by 255 for visualization
    print(f"Saved: {save_path}")
    return mask

# ---------------- RUN INFERENCE ----------------
if os.path.isfile(IMG_DIR):
    # single image
    predict_mask(IMG_DIR)
else:
    # folder of images
    for img_file in os.listdir(IMG_DIR):
        if img_file.lower().endswith((".jpg", ".png")):
            predict_mask(os.path.join(IMG_DIR, img_file))
