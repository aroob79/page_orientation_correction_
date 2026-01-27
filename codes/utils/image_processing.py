import cv2
import numpy as np

def enhance_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_img = cv2.merge((l, a, b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(enhanced_img, 9, 75, 75)

def apply_perspective_correction(image, box):
    rect = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    rect[0] = box[np.argmin(s)] # TL
    rect[2] = box[np.argmax(s)] # BR
    
    diff = np.diff(box, axis=1)
    rect[1] = box[np.argmin(diff)] # TR
    rect[3] = box[np.argmax(diff)] # BL

    (tl, tr, br, bl) = rect

    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(height_left), int(height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 270 or angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image

def rotate_image2(img, angle):
    """
    Rotate image by given angle (counter-clockwise) around center
    """
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    
    
    # Get rotation matrix (positive angle = counter-clockwise)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image size to prevent cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new size
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform rotation with white background
    rotated = cv2.warpAffine(img, M, (new_w, new_h), 
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    return rotated


def read_image_any(img_path: str):
    # 1) Try OpenCV first (fast; supports most common formats)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img

    # 2) Fallback to Pillow (supports more formats; can be extended with plugins)
    try:
        from PIL import Image

        # Optional: enable HEIC/HEIF support if installed
        # pip install pillow-heif
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except Exception:
            pass

        with Image.open(img_path) as im:
            # Normalize modes
            if im.mode in ("P", "LA"):
                im = im.convert("RGBA")
            elif im.mode != "RGB" and im.mode != "RGBA":
                im = im.convert("RGB")

            arr = np.array(im)

            # Convert to OpenCV-friendly format (BGR/BGRA)
            if arr.ndim == 2:
                return arr  # grayscale
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    except Exception:
        return None
