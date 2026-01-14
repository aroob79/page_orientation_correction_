import cv2
import numpy as np

def get_largest_component(mask):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 1
    return largest_mask

def fill_and_split_mask(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    dist_transform = cv2.distanceTransform(filled_mask, cv2.DIST_L2, 5)
    _, last_seeds = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    last_seeds = np.uint8(last_seeds)

    num_labels, markers = cv2.connectedComponents(last_seeds)
    markers = markers + 1
    unknown = cv2.subtract(filled_mask, last_seeds)
    markers[unknown == 255] = 0

    img_color = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    
    part1, part2 = np.zeros_like(mask), np.zeros_like(mask)
    part1[markers == 2], part2[markers == 3] = 255, 255

    return part1, part2, (part1 if np.sum(part1) > np.sum(part2) else part2)

def get_regular_rectangular_mask(mask, scale_factor=0.95):
    mask = mask.astype(np.uint8)
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    (center_x, center_y), (width, height), angle = rect
    shrunk_rect = ((center_x, center_y), (width * scale_factor, height * scale_factor), angle)
    
    box = cv2.boxPoints(shrunk_rect)
    box[:, 0] = np.clip(box[:, 0], 0, w - 1)
    box[:, 1] = np.clip(box[:, 1], 0, h - 1)
    final_box = box.astype(np.int64)

    rect_mask = np.zeros_like(mask)
    cv2.fillPoly(rect_mask, [final_box], 1)
    return rect_mask, final_box