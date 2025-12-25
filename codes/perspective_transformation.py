import cv2
import numpy as np
from matplotlib import pyplot as plt

def order_points(pts):
    """
    Order points in the order: top-left, top-right, bottom-right, bottom-left
    """
    # Initialize array to hold ordered coordinates
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    # Top-left will have smallest sum
    rect[0] = pts[np.argmin(s)]
    # Bottom-right will have largest sum
    rect[2] = pts[np.argmax(s)]
    # Top-right will have smallest difference
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have largest difference
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_point_transform(image, pts):
    """
    Apply perspective transformation given 4 corner points
    
    Parameters:
    - image: input image
    - pts: array of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
           (order doesn't matter, will be auto-ordered)
    
    Returns:
    - warped: perspective-corrected image
    """
    # Order the points: top-left, top-right, bottom-right, bottom-left
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate width of new image
    # Take maximum of top and bottom edge lengths
    widthA = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    widthB = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    maxWidth = max(int(widthA), int(widthB))
    
    # Calculate height of new image
    # Take maximum of left and right edge lengths
    heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points for the transform (perfect rectangle)
    dst = np.array([
        [0, 0],                          # Top-left
        [maxWidth - 1, 0],               # Top-right
        [maxWidth - 1, maxHeight - 1],   # Bottom-right
        [0, maxHeight - 1]               # Bottom-left
    ], dtype="float32")
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped, M, rect


def correct_perspective_with_points(image_path, corner_points,is_visual = True):
    """
    Apply perspective correction given 4 corner points
    
    Parameters:
    - image_path: path to image
    - corner_points: numpy array of shape (4, 2) with corner coordinates
                     [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                     Order doesn't matter
    
    Returns:
    - corrected: perspective-corrected image
    """
    # Read image
    img = image_path
    
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Convert corner_points to numpy array if not already
    corners = np.array(corner_points, dtype="float32")
    
    # Validate input
    if corners.shape != (4, 2):
        print(f"Error: corner_points must be shape (4, 2), got {corners.shape}")
        return None
    
    # Apply perspective transform
    corrected, transform_matrix, ordered_corners = four_point_transform(img, corners)
    
    print("\n" + "="*80)
    print("PERSPECTIVE CORRECTION")
    print("="*80)
    print("\nOrdered Corner Points (TL, TR, BR, BL):")
    labels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
    for i, (label, corner) in enumerate(zip(labels, ordered_corners)):
        print(f"  {label:12}: ({corner[0]:.1f}, {corner[1]:.1f})")
    
    print(f"\nOriginal image size: {img.shape[1]} x {img.shape[0]}")
    print(f"Corrected image size: {corrected.shape[1]} x {corrected.shape[0]}")
    print("="*80 + "\n")
    
    # Visualize 
    if is_visual:
        visualize_correction(img, corrected, ordered_corners)
    
    return corrected, transform_matrix, ordered_corners


def visualize_correction(original, corrected, corners):
    """
    Visualize the perspective correction process
    """
    img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    
    # ============ FIGURE 1: Original with corners marked ============
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    img_with_corners = img_rgb.copy()
    
    # Draw corners
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    labels = ['TL', 'TR', 'BR', 'BL']
    
    for i, (corner, color, label) in enumerate(zip(corners, colors, labels)):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(img_with_corners, (x, y), 15, color, -1)
        cv2.putText(img_with_corners, label, (x+25, y+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    # Draw lines connecting corners
    for i in range(4):
        pt1 = tuple(corners[i].astype(int))
        pt2 = tuple(corners[(i+1)%4].astype(int))
        cv2.line(img_with_corners, pt1, pt2, (0, 255, 255), 3)
    
    axes[0].imshow(img_with_corners)
    axes[0].set_title('Original Image with Corner Points', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(corrected_rgb)
    axes[1].set_title('Perspective Corrected', fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('perspective_correction.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ============ FIGURE 2: Before and After Comparison ============
    fig, axes = plt.subplots(2, 1, figsize=(16, 16))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('BEFORE: Distorted Perspective', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(corrected_rgb)
    axes[1].set_title('AFTER: Corrected (Bird\'s Eye View)', fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('before_after_correction.png', dpi=150, bbox_inches='tight')
    plt.show() 



