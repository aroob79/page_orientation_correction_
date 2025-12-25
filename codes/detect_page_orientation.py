from typing import Any


import cv2 
import os  
from ultralytics import YOLO
import numpy as  np 
from pathlib import Path
import matplotlib.pyplot as plt 
import random

from page_rotation import rotate_image, detect_and_visualize_text_lines 
from find_corner_point import CornerPointFinder 
from perspective_transformation import correct_perspective_with_points 
from filter_img_to_super_res import DocumentEnhancer 

## first load the image corner detector 

class PageOrientationDetector:
    def __init__(self, corner_model_path: str, sr : '2 | 4' = 4, apply_enhancement: bool =True):
        
        ## first loading the model
        self.model  =  self.load_model(corner_model_path) 
        self.class_names =  self.model.names
        self.finder = CornerPointFinder()
        self.sr: int = sr
        self.apply_enhancement = apply_enhancement
        if self.apply_enhancement:
            self.enhancer = DocumentEnhancer(apply_sr=True, sr_scale=self.sr)
        
       
        
    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # -------------------------------------------------
        # 1️⃣ Load trained YOLO segmentation model
        # -------------------------------------------------
        model = YOLO(model_path)  # path to your trained weights

        return model 
    
    def prediction_(self, image_path: str,conf_threshold: float =0.3, 
                 iou_threshold: float =0.7):
        """
        Predict page orientation using the loaded YOLO model
        """
        
        results = self.model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        save=False,
        verbose=False
        )
        return results[0]
    

    
    def plot_mask_coordinates_overlay(self, image, masks_xy,classes, save_path=None):
        """
        Plot mask coordinates as scatter points overlaid on the original image.
        
        Args:
            image: Original image (numpy array or path)
            results: YOLO results object
            save_path: Optional path to save the plot
        """
        
        # Handle image input
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(image)
       
        
    
        
        # Get mask coordinate
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Show image
        ax.imshow(img)
        
        # Plot each mask
        for i, (mask_coords, cls) in enumerate(zip(masks_xy, classes)):
            if len(mask_coords) > 0:
                x_coords = mask_coords[:, 0]
                y_coords = mask_coords[:, 1]
                
                # Get color for this class
                color_rgb = tuple(c/255.0 for c in self.colors[cls])
                
                # Plot scatter
                ax.scatter(x_coords, y_coords, 
                          c=[color_rgb], 
                          s=3, 
                          alpha=0.8,
                          edgecolors='red',
                          linewidths=0.5,
                          label=f"{self.class_names[cls]} (#{i+1})")
        
        ax.set_xlabel('X coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y coordinate (pixels)', fontsize=12)
        ax.set_title('Mask Coordinates Overlay on Image', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save or show
        if save_path is not None:
            plt.savefig(os.path.join(save_path,'boundary_overlay.jpg'), dpi=150, bbox_inches='tight')
            print(f"Saved overlay plot to: {save_path}")
        else:
            plt.show()
        
        return fig, ax 
    def calculate_polygon_area(self,points):
        """Calculate area of polygon using Shoelace formula"""
        points = np.array(points)
        n = len(points)
        area = 0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2
    
    def main(self, image_path: str,  save_paths:str=None, is_vizualize: bool =True,conf_threshold: float =0.3, 
                 iou_threshold: float =0.7):
        """
        Main function to run detection and save results.
        """
        ##first find the rotation and correct the rotation 
        rotation_angle, corrected_img = detect_and_visualize_text_lines(image_path,save_paths)
        print(f"Detected rotation angle: {rotation_angle:.2f}°")
        
        results_ = self.prediction_(corrected_img,conf_threshold, 
                 iou_threshold)
        
        # # print(type(corrected_img))
        # if  is_vizualize:
       
    
        #     masks_xy = results_.masks.xy
        #     classes = results_.boxes.cls.cpu().numpy().astype(int)
                
        #     self.plot_mask_coordinates_overlay(self, corrected_img, masks_xy,classes)
           
        # if save_paths is not None:
        #     self.plot_mask_coordinates_overlay(self, corrected_img, masks_xy,classes, save_path=save_paths)
        
        ## 
        image = corrected_img #cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)
    
        # Get mask coordinates from YOLO results
        # mask_coords should be numpy array of shape (N, 2)
        # Example: 
        mask_coords = results_.masks.xy[0]
        # Method 1
        corners1 = self.finder.method1_convex_hull_extreme(mask_coords)
        print("Method 1 - Convex Hull Extreme Points:")
        
        
        polygon_area = self.calculate_polygon_area(corners1)
        img_area = image.shape[0] * image.shape[1]
        print(polygon_area/img_area)
        threshold = 0.15
        if polygon_area < (threshold * img_area):
            print(f"Warning: Detected polygon area is less than 75% of image area. Using different way.")
            corners1 = self.finder.method3_polygon_approximation(mask_coords)
            print("\nMethod 3 - Polygon Approximation:")
        
        if is_vizualize:
            self.finder.visualize_corners(image, mask_coords, corners1, 
                        "Method 3: Polygon Approx", save_paths) 
            
        polygon_area = self.calculate_polygon_area(corners1)
        print(polygon_area/img_area)
        if polygon_area > (threshold * img_area): 
            image, matrix, ordered_corners = correct_perspective_with_points(image, corners1,is_vizualize)
        if save_paths is not None:
            cv2.imwrite(os.path.join(save_paths,'corrected_page.jpg'), image)
            
        condition  = results_.names[int(results_.boxes.cls[0])]
        print(f"Detected page orientation: {condition}")
        if condition == 'down':
            image = rotate_image(image, 180) 
            if save_paths is not None:
                cv2.imwrite(os.path.join(save_paths,'fliped_corrected_page.jpg'), image) 

        if self.apply_enhancement:
            if self.sr in [2,4]:
                image = self.enhancer.enhance(image)
            else:
                print(f"Invalid super-resolution scale: {self.sr}. Skipping enhancement.")   
        return image   


    def process_and_save(
        self,
        input_path: str,
        output_dir: str,
        save_paths: str = None,
        is_visualize: bool = False,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.7,

    ):
        """
        Process a single image or a folder of images and save outputs.

        Rules:
        - If input_path is an image → process & save with same name
        - If input_path is a folder:
            - If save_paths is provided → randomly select ONE image and save it
            - Else → process & save ALL images
        """

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_paths, exist_ok=True) if save_paths else None

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        # -----------------------------
        # Case 1: Single Image
        # -----------------------------
        if os.path.isfile(input_path):
            if not input_path.lower().endswith(valid_exts):
                raise ValueError(f"Unsupported image format: {input_path}")

            img_name = os.path.basename(input_path)
            output_path = os.path.join(output_dir, img_name)

            result_img = self.main(
                image_path=input_path,
                save_paths=save_paths,
                is_vizualize=is_visualize,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

            cv2.imwrite(output_path, result_img)
            print(f"✅ Saved output: {output_path}")
            return

        # -----------------------------
        # Case 2: Folder of Images
        # -----------------------------
        if os.path.isdir(input_path):
            images = [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if f.lower().endswith(valid_exts)
            ]

            if not images:
                raise ValueError("No valid images found in the folder.")

            # 🎯 Special condition: randomly select ONE image
            if save_paths is not None:
                selected_image = random.choice(images)
                img_name = os.path.basename(selected_image)
                output_path = os.path.join(output_dir, img_name)

                result_img = self.main(
                    image_path=selected_image,
                    save_paths=save_paths,
                    is_vizualize=is_visualize,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    
                )

                cv2.imwrite(output_path, result_img)
                print(f"🎯 Randomly selected & saved: {output_path}")
                return

            # 🔁 Otherwise process ALL images
            for img_path in images:
                img_name = os.path.basename(img_path)
                output_path = os.path.join(output_dir, img_name)

                result_img = self.main(
                    image_path=img_path,
                    save_paths=None,
                    is_vizualize=is_visualize,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                     
                )

                cv2.imwrite(output_path, result_img)
                print(f"✅ Saved: {output_path}")

            return

        raise ValueError("Input path must be a valid image or directory.")

    
        
        
    
if __name__ == "__main__":
    model_path  = "/mnt/storage1/workspace/arobin/page_orientation/models/best.pt"
    image_path = '/mnt/storage1/workspace/arobin/page_orientation/test_img/photo_6321007273832024894_y.jpg'
    save_dir = "/mnt/storage1/workspace/arobin/page_orientation/output_images1"
    detector = PageOrientationDetector(model_path,sr=4,apply_enhancement=True)
    print(f"Model loaded successfully. {detector.model.names}")
    detector.process_and_save(
    input_path=image_path,
    output_dir=save_dir,
    save_paths="/mnt/storage1/workspace/arobin/page_orientation/viz_outputs",
    
)