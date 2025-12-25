import cv2
import numpy as np
import os 

def rotate_image(img, angle):
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

def detect_and_visualize_text_lines(image_path,save_dir=None):
    """
    Detect text lines using Hough Line Transform and visualize only the detected lines
    """

    os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None  
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=100, maxLineGap=10)
    
    # Store and categorize lines
    text_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if ((x1**2 + y1 **2 ) ** 0.5) > ((x2**2 + y2 **2 ) ** 0.5):
                x1, y1, x2, y2 = x2, y2, x1, y1
            
            # Calculate angle
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Normalize to [-90, 90]
            if angle < -90:
                angle += 180
            elif angle > 90:
                angle -= 180
            
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            line_data = {
                'coords': (x1, y1, x2, y2),
                'angle': angle,
                'length': length
            }
            
            
            text_lines.append(line_data)
        
    
    # Sort text lines by Y coordinate (top to bottom)
    # text_lines.sort(key=lambda l: (l['coords'][1] + l['coords'][3]) / 2)
    all_angles = [i['angle'] for i in text_lines] 
    # Suppose arr is your array
    counts, bins = np.histogram(all_angles, bins=30)

    # Index of bin with the max count
    max_bin_index = np.argmax(counts)

    # Get bin range
    max_bin_start = bins[max_bin_index]
    max_bin_end = bins[max_bin_index + 1]
    text_lines = [i for i in text_lines if max_bin_start <= i['angle'] <= max_bin_end]
    
    angles = [l['angle'] for l in text_lines]
    median_angle = np.median(angles) 
    
    rotation_needed = median_angle 
    # rotate image if needed
    if abs(rotation_needed) > 0.1:
        img_rotate = rotate_image(img, rotation_needed)
        img_rgb = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2RGB)  
    if save_dir is not None:
        cv2.imwrite(os.path.join(save_dir,'rotated_image.jpg'), img_rgb)
        
    return rotation_needed, img_rgb 


if __name__ == "__main__":
    # Replace with your image path
    image_path = '/mnt/storage1/workspace/arobin/page_orientation/test_img/photo_6321007273832024894_y.jpg'
    
    try:
        rotation_angle, corrected_img = detect_and_visualize_text_lines(image_path, save_dir='./temp_output')
        print(f"Detected rotation angle: {rotation_angle:.2f}°")
        
        # Visualize corrected image
        import matplotlib.pyplot as plt
        plt.imshow(corrected_img)
        plt.title(f'Corrected Image (Rotated by {rotation_angle:.2f}°)')
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")