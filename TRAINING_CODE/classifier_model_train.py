import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier 
import joblib

def extract_hog(img):
    img = cv2.resize(img, (128,128))
    return hog(img, orientations=9,
               pixels_per_cell=(8,8),
               cells_per_block=(2,2),
               block_norm='L2-Hys')

X, y = [], []

paths = ['/content/drive/MyDrive/Ocr_img/yolo_sorted_by_class/class_0','/content/drive/MyDrive/Ocr_img/yolo_sorted_by_class/class_1']
for path in paths:
  for img_path in os.listdir(path):
    if img_path.endswith('.jpg'):
      full_path = os.path.join(path,img_path)

      img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
      X.append(extract_hog(img))
      if path == paths[0]:
        y.append(0)
      else:
        y.append(1)

X = np.array(X)
y = np.array(y)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,stratify=y)


clf = XGBClassifier(

)

clf.fit(Xtr, ytr)

print("Accuracy:", clf.score(Xte, yte))

config = {
    "img_size": 128,
    "hog": {
        "orientations": 9,
        "pixels_per_cell": (8,8),
        "cells_per_block": (2,2)
    }
}

joblib.dump((clf, config), "/content/drive/MyDrive/Ocr_img/models/rf_text_type_model.pkl")
print("Model saved.")