import numpy as np
import config 

def calculate_polygon_area(points):
        """Calculate area of polygon using Shoelace formula"""
        points = np.array(points)
        n = len(points)
        area = 0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2

def predict_page_type(img_path, model):
    import cv2
    from skimage.feature import hog
    import numpy as np

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128,128))

    feat = hog(
        img,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )

    feat = feat.reshape(1, -1)
    pred = model.predict(feat)[0]

    return "up" if pred == 1 else "down"