import os
import cv2
import torch
import numpy as np
import config
from models.model_loader import load_segmentation_model,load_model,load_classifier_model
from utils.image_processing import enhance_image, apply_perspective_correction, rotate_image,rotate_image2
from utils.mask_processing import (
    get_largest_component, 
    fill_and_split_mask, 
    get_regular_rectangular_mask
)
from utils.orientation_detector import determine_orientation
from find_corner_point import CornerPointFinder 
from utils.utils_ import calculate_polygon_area,predict_page_type
from filter_img_to_super_res import DocumentEnhancer

enhancer = DocumentEnhancer(apply_sr=config.APPLY_SR, sr_scale=config.SR_SCALE)

# Load Global Model
model = load_segmentation_model()
# yolo_model = load_model(config.YOLO_MODEL_PATH)
classifier_model, classifier_config = load_classifier_model(config.CLASSIFIER_MODEL_PATH)  

config.HOG_CONFIG = classifier_config
# conf_threshold = config.YOLO_CONFIDENCE_THRESHOLD
# iou_threshold = config.YOLO_IOU_THRESHOLD # You can adjust this value as needed
# finder = CornerPointFinder()    

def process_image(img_path):
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None: return

    # Preprocessing
    enhanced = enhance_image(img)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (config.IMG_SIZE, config.IMG_SIZE)) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_norm = (img_resized - mean) / std
    tensor = torch.tensor(img_norm).permute(2,0,1).unsqueeze(0).float().to(config.DEVICE)

    # Inference
    with torch.no_grad():
        out = model(tensor)["out"]
        mask = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Post-processing
    mask_rescaled = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    largest_region = get_largest_component(mask_rescaled)
    _, _, final_split_mask = fill_and_split_mask(largest_region)
    filtered_mask, box = get_regular_rectangular_mask(final_split_mask, config.SCALE_FACTOR)

    # Perspective Correction & Saving
    if box is not None:
        corrected_page = apply_perspective_correction(img, box)

        # Step B: Detect Orientation (Check if landscape/portrait is wrong)
        rotation_needed = determine_orientation(corrected_page)
        if rotation_needed != 0:
            corrected_page = rotate_image(corrected_page, rotation_needed)
            print(f"Rotated {img_name} by {rotation_needed} degrees.")

        if config.IS_SAVE_INTERMEDIATE:
            save_path_rect = os.path.join(config.SAVE_DIR, img_name.replace(".jpg", "_corrected.jpg"))
            cv2.imwrite(save_path_rect, corrected_page)
            
            # Save Debug Bbox Image
            cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
            cv2.imwrite(os.path.join(config.SAVE_DIR, img_name.replace(".jpg", "_bbox.jpg")), img)

    # ## predicting from yolo model 
    # results = yolo_model.predict(
    #                 source=corrected_page,
    #                 conf=conf_threshold,
    #                 iou=iou_threshold,
    #                 save=False,
    #                 verbose=False
    #                 )[0]
    # print(f"YOLO Results: {results.masks}")
    # if results.masks is not None and len(results.masks) > 0:
    #     mask_coords = results.masks.xy[0]
    #     corners1 = finder.method1_convex_hull_extreme(mask_coords)
    #     print("Method 1 - Convex Hull Extreme Points:")

    #     polygon_area = calculate_polygon_area(corners1)
    #     img_area = corrected_page.shape[0] * corrected_page.shape[1]
    #     print(polygon_area/img_area)
    #     threshold = 0.15
    #     if polygon_area < (threshold * img_area):
    #         print(f"Warning: Detected polygon area is less than 75% of image area. Using different way.")
    #         corners1 =  finder.method3_polygon_approximation(mask_coords)
    #         print("\nMethod 3 - Polygon Approximation:")
            
    #         if corners1 is None:
    #             print("Could not find suitable corners using Method 3. Skipping perspective correction.")
    #         else:
    #             polygon_area = calculate_polygon_area(corners1)
    #             print(polygon_area/img_area)
    #             if polygon_area > (threshold * img_area): 
    #                 corrected_page = apply_perspective_correction(corrected_page, corners1)

    # condition  = results.names[int(results.boxes.cls[0])]
    condition = predict_page_type(img_path, classifier_model)   
    print(f"Detected page orientation: {condition} page name {img_name}")
    if condition == 'down':
        corrected_page = rotate_image2(corrected_page, 180) 

    if config.APPLY_SR:
        if config.SR_SCALE in [2,4]:
            corrected_page = enhancer.enhance(corrected_page)
        else:
            print(f"Invalid super-resolution scale: {config.SR_SCALE}. Skipping enhancement.") 

    save_path_final = os.path.join(config.SAVE_DIR, img_name.replace(".jpg", "_corrected2.jpg"))
    cv2.imwrite(save_path_final, corrected_page)      
    return corrected_page   
        

    print(f"Processed: {img_name}")

if __name__ == "__main__":
    if os.path.isfile(config.IMG_DIR):
        process_image(config.IMG_DIR)
    else:
        for file in os.listdir(config.IMG_DIR):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                process_image(os.path.join(config.IMG_DIR, file))