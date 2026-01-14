import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

def parse_yolo_label(label_path):
    """Extract class IDs from a YOLO label file."""
    classes = set()
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    except:
        pass
    return classes

def split_yolo_dataset(images_dir, labels_dir, output_dir, val_samples_per_class=50, seed=42):
    """
    Split YOLO segmentation dataset with balanced validation set.
    
    Args:
        images_dir: Path to folder containing images
        labels_dir: Path to folder containing label files
        output_dir: Path to output directory
        val_samples_per_class: Number of samples per class in validation set
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create output directories
    output_path = Path(output_dir)
    train_img_dir = output_path / 'images'/ 'train' 
    train_lbl_dir = output_path / 'labels' / 'train'
    val_img_dir = output_path / 'images'/ 'val' 
    val_lbl_dir = output_path / 'labels'/ 'val' 
    
    for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Group images by their classes
    class_to_images = defaultdict(list)
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    print("Analyzing dataset...")
    for img_file in images_path.iterdir():
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Find corresponding label file
            label_file = labels_path / f"{img_file.stem}.txt"
            
            if label_file.exists():
                classes = parse_yolo_label(label_file)
                
                # Add image to each class it contains
                for cls in classes:
                    class_to_images[cls].append(img_file.name)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    for cls, images in sorted(class_to_images.items()):
        print(f"Class {cls}: {len(images)} images")
    
    # Select validation samples - ensure equal representation
    val_images = set()
    
    print(f"\nSelecting {val_samples_per_class} samples per class for validation...")
    for cls in sorted(class_to_images.keys()):
        available = class_to_images[cls]
        
        if len(available) < val_samples_per_class:
            print(f"Warning: Class {cls} has only {len(available)} images, using all of them")
            selected = available
        else:
            selected = random.sample(available, val_samples_per_class)
        
        val_images.update(selected)
        print(f"Class {cls}: Selected {len(selected)} images for validation")
    
    # Get all images for train set (excluding val images)
    all_images = set(img.name for img in images_path.iterdir() 
                     if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
    train_images = all_images - val_images
    
    print(f"\nTotal validation images: {len(val_images)}")
    print(f"Total training images: {len(train_images)}")
    
    # Copy files to train directory
    print("\nCopying training files...")
    for img_name in train_images:
        img_src = images_path / img_name
        lbl_src = labels_path / f"{Path(img_name).stem}.txt"
        
        if lbl_src.exists():
            shutil.copy2(img_src, train_img_dir / img_name)
            shutil.copy2(lbl_src, train_lbl_dir / f"{Path(img_name).stem}.txt")
    
    # Copy files to val directory
    print("Copying validation files...")
    for img_name in val_images:
        img_src = images_path / img_name
        lbl_src = labels_path / f"{Path(img_name).stem}.txt"
        
        if lbl_src.exists():
            shutil.copy2(img_src, val_img_dir / img_name)
            shutil.copy2(lbl_src, val_lbl_dir / f"{Path(img_name).stem}.txt")
    
    # Verify validation set balance
    print("\nValidation set class distribution:")
    val_class_count = defaultdict(int)
    for img_name in val_images:
        lbl_path = val_lbl_dir / f"{Path(img_name).stem}.txt"
        classes = parse_yolo_label(lbl_path)
        for cls in classes:
            val_class_count[cls] += 1
    
    for cls in sorted(val_class_count.keys()):
        print(f"Class {cls}: {val_class_count[cls]} images in validation set")
    
    print("\nDataset split complete!")
    print(f"Output directory: {output_dir}")

# Example usage
if __name__ == "__main__":
    # Update these paths to your dataset
    IMAGES_DIR = "/mnt/storage1/workspace/arobin/page_orientation/data/PAGE_ORIENTATION_DATA_WITH_4_CLASS/images"
    LABELS_DIR = "/mnt/storage1/workspace/arobin/page_orientation/data/PAGE_ORIENTATION_DATA_WITH_4_CLASS/labels"
    OUTPUT_DIR = "/mnt/storage1/workspace/arobin/page_orientation/data/splited_data_with_4_class"
    
    # Number of samples per class in validation set
    VAL_SAMPLES_PER_CLASS = 20
    
    split_yolo_dataset(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        val_samples_per_class=VAL_SAMPLES_PER_CLASS,
        seed=42
    )