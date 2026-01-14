import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks

def get_structural_score(profile):
    """Analyzes the signal to distinguish structured text from noise."""
    # 1. Normalize
    norm = (profile - np.min(profile)) / (np.ptp(profile) + 1e-6)
    
    # 2. Dynamic Peak Detection
    peaks, _ = find_peaks(norm, prominence=0.3)
    num_peaks = len(peaks)
    
    # 3. Calculate "Valley Depth" (Percentage of signal near the floor)
    low_regions = np.sum(norm < 0.3) / len(norm)
    
    # 4. Signal Swing (Standard Deviation)
    swing = np.std(norm)
    
    # Score: High peaks + clear valleys + high variance = likely text lines
    score = num_peaks * low_regions * swing
    return score

def determine_orientation(image):
    """
    Returns 0 if the text is horizontal (rows are structured), 
    or 90 if the text is vertical (columns are structured).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect text characters into lines
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Vertical Projection (Column-wise sum)
    col_hist = np.sum(dilated, axis=0)
    col_hist = savgol_filter(col_hist / (col_hist.max() + 1e-6), 31, 3)

    # Horizontal Projection (Row-wise sum)
    row_hist = np.sum(dilated, axis=1)
    row_hist = savgol_filter(row_hist / (row_hist.max() + 1e-6), 31, 3)

    col_score = get_structural_score(col_hist)
    row_score = get_structural_score(row_hist)

    # If row_score is higher, peaks are found across rows (Standard Horizontal Text)
    # If col_score is higher, the image likely needs a 90-degree rotation
    if row_score >= col_score:
        return 0  # Already correct
    else:
        return 90 # Needs rotation