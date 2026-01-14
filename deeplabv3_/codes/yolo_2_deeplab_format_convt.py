import cv2
import numpy as np
import os

def yolo_to_binary_mask(img_path, label_path, out_mask_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)  # 0 = background

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            data = line.strip().split()
            # YOLO-seg: class x1 y1 x2 y2 ...
            coords = list(map(float, data[1:]))
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i+1] * h)
                points.append([x, y])
            pts = np.array(points, np.int32)
            cv2.fillPoly(mask, [pts], 1)  # 1 = foreground

    cv2.imwrite(out_mask_path, mask*255)  # save as 0/255



img_dir = "/mnt/storage1/workspace/arobin/bangla_printed_ocr/bangla_ocr/page_orientation/using_deeplabv3/data/raw_data/images"
label_dir = "/mnt/storage1/workspace/arobin/bangla_printed_ocr/bangla_ocr/page_orientation/using_deeplabv3/data/raw_data/labels"
mask_dir = "/mnt/storage1/workspace/arobin/bangla_printed_ocr/bangla_ocr/page_orientation/using_deeplabv3/data/raw_data/masks"

os.makedirs(mask_dir, exist_ok=True)

num_classes = 2  # change if needed

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
    out_mask = os.path.join(mask_dir, img_name.replace(".jpg", ".png"))

    yolo_to_binary_mask(img_path, label_path, out_mask_path=out_mask)
print(f"Processed {img_name}")