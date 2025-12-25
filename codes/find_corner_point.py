import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os 


class CornerPointFinder:
    """Find 4 corner points from segmentation mask coordinates."""
    
    @staticmethod
    def method1_convex_hull_extreme(mask_coords):
        """
        Method 1: Find extreme points from convex hull.
        Works well for rectangular or quadrilateral shapes.
        
        Args:
            mask_coords: Numpy array of shape (N, 2) with [x, y] coordinates
            
        Returns:
            corners: Numpy array of shape (4, 2) with corner coordinates
                    Order: [top-left, top-right, bottom-right, bottom-left]
        """
        if len(mask_coords) < 4:
            return None
        
        # Compute convex hull
        hull = cv2.convexHull(mask_coords.astype(np.float32))
        hull = hull.reshape(-1, 2)
        
        # Find extreme points
        top_left = hull[np.argmin(hull[:, 0] + hull[:, 1])]  # Min x+y
        top_right = hull[np.argmin(hull[:, 1] - hull[:, 0])]  # Min y-x
        bottom_right = hull[np.argmax(hull[:, 0] + hull[:, 1])]  # Max x+y
        bottom_left = hull[np.argmax(hull[:, 1] - hull[:, 0])]  # Max y-x
        
        corners = np.array([top_left, top_right, bottom_right, bottom_left])
        return corners
    
    
    @staticmethod
    def method3_polygon_approximation(mask_coords, epsilon_factor=0.02):
        """
        Method 3: Polygon approximation to reduce to 4 points.
        Uses Douglas-Peucker algorithm.
        
        Args:
            mask_coords: Numpy array of shape (N, 2) with [x, y] coordinates
            epsilon_factor: Approximation accuracy factor (0.01-0.05 typical)
            
        Returns:
            corners: Numpy array of shape (4, 2) with corner coordinates
        """
        if len(mask_coords) < 4:
            return None
        
        # Calculate epsilon as a percentage of the perimeter
        perimeter = cv2.arcLength(mask_coords.astype(np.float32), True)
        epsilon = epsilon_factor * perimeter
        
        # Approximate polygon
        approx = cv2.approxPolyDP(mask_coords.astype(np.float32), epsilon, True)
        approx = approx.reshape(-1, 2)
        
        # If we get exactly 4 points, return them
        if len(approx) == 4:
            corners = CornerPointFinder._order_corners(approx)
            return corners
        
        # If more than 4, use method 2 as fallback
        elif len(approx) > 4:
            return CornerPointFinder.method2_bounding_rect(mask_coords)
        
        # If less than 4, return None
        else:
            return None
    
    @staticmethod
    def _order_corners(corners):
        """
        Order corners in clockwise order starting from top-left.
        Order: [top-left, top-right, bottom-right, bottom-left]
        
        Args:
            corners: Numpy array of shape (4, 2)
            
        Returns:
            ordered_corners: Numpy array of shape (4, 2)
        """
        # Calculate centroid
        centroid = np.mean(corners, axis=0)
        
        # Calculate angles from centroid
        angles = np.arctan2(corners[:, 1] - centroid[1], 
                           corners[:, 0] - centroid[0])
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]
        
        # Find top-left (minimum y+x)
        top_idx = np.argmin(sorted_corners[:, 0] + sorted_corners[:, 1])
        
        # Rotate array so top-left is first
        ordered_corners = np.roll(sorted_corners, -top_idx, axis=0)
        
        return ordered_corners
    
    @staticmethod
    def visualize_corners(image, mask_coords, corners, method_name="", 
                         save_path=None, show_mask=True):
        """
        Visualize the found corner points on the image.
        
        Args:
            image: Original image (numpy array)
            mask_coords: Original mask coordinates
            corners: Found corner points (4, 2)
            method_name: Name of the method used
            save_path: Path to save the visualization
            show_mask: Whether to show all mask points
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Display image
        ax.imshow(image)
        
        # Plot mask coordinates
        if show_mask:
            ax.scatter(mask_coords[:, 0], mask_coords[:, 1], 
                      c='cyan', s=1, alpha=0.3, label='Mask points')
        
        # Plot corner points
        if corners is not None:
            corner_labels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
            colors = ['red', 'green', 'blue', 'yellow']
            
            for i, (corner, label, color) in enumerate(zip(corners, corner_labels, colors)):
                ax.scatter(corner[0], corner[1], c=color, s=200, 
                          marker='o', edgecolors='white', linewidths=2,
                          label=f'{label}', zorder=5)
                ax.annotate(f'{i+1}', (corner[0], corner[1]), 
                           fontsize=12, color='white', weight='bold',
                           ha='center', va='center')
            
            # Draw lines connecting corners
            corners_closed = np.vstack([corners, corners[0]])
            ax.plot(corners_closed[:, 0], corners_closed[:, 1], 
                   'r--', linewidth=2, alpha=0.7, label='Corner connections')
        
        ax.set_title(f'Corner Detection: {method_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path,'with_corner_point.jpg'), dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.show()
        
        return fig, ax




# Example usage
if __name__ == "__main__":
    # Assume you have results from YOLO inference
    # results = model.predict(image)
    # mask_coords = results.masks.xy[0]  # First detected object
    
    # Example: Create dummy data for demonstration
    # Replace this with your actual mask coordinates
    import cv2
    
    # Load your image
    image = corrected_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get mask coordinates from YOLO results
    # mask_coords should be numpy array of shape (N, 2)
    # Example: 
    mask_coords = results.masks.xy[0]
    
    
    # Find corners using different methods
    finder = CornerPointFinder()
    
    # Try different methods
    print("Testing different corner detection methods:\n")
    
    # Method 1
    corners1 = finder.method1_convex_hull_extreme(mask_coords)
    print("Method 1 - Convex Hull Extreme Points:")
    print(corners1)
    finder.visualize_corners(image, mask_coords, corners1, 
                            "Method 1: Convex Hull", "corners_method1.png")
    
    
    # Method 3
    corners3 = finder.method3_polygon_approximation(mask_coords)
    print("\nMethod 3 - Polygon Approximation:")
    print(corners3)
    finder.visualize_corners(image, mask_coords, corners3, 
                            "Method 3: Polygon Approx", "corners_method3.png")
    
   
    