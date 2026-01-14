import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

class BinarySegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.imgs = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Image
        img = cv2.imread(os.path.join(self.img_dir, self.imgs[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = img / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = torch.tensor(img).permute(2,0,1).float()

        # Mask
        mask_path = os.path.join(self.mask_dir, self.imgs[idx].replace(".jpg", ".png"))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)  # Convert any non-zero to 1
        mask = torch.tensor(mask, dtype=torch.long)

        return img, mask
