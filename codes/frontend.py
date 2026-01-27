import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
import tempfile
import shutil
import io

# ============================================================
# Image Format Support
# ============================================================
# Try to import pillow-heif for HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
    print("✓ HEIC/HEIF support enabled")
except ImportError:
    HEIC_SUPPORTED = False
    print("⚠ HEIC support not available. Install with: pip install pillow-heif")

# Supported image extensions
SUPPORTED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif',
    '.webp', '.ico', '.ppm', '.pgm', '.pbm', '.pnm'
}

if HEIC_SUPPORTED:
    SUPPORTED_EXTENSIONS.update({'.heic', '.heif', '.avif'})


def convert_to_cv2_image(file_path):
    """
    Convert any supported image format to CV2 BGR numpy array.
    Handles HEIC, WebP, and other formats through PIL.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # For standard formats, try cv2 first (faster)
        if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
            img = cv2.imread(file_path)
            if img is not None:
                return img
        
        # For WebP, HEIC, HEIF, AVIF and fallback for other formats, use PIL
        pil_image = Image.open(file_path)
        
        # Handle different color modes
        if pil_image.mode == 'RGBA':
            # Convert RGBA to RGB with white background
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])
            pil_image = background
        elif pil_image.mode == 'P':
            # Convert palette mode to RGB
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode == 'L':
            # Convert grayscale to RGB
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode != 'RGB':
            # Convert any other mode to RGB
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to numpy array (RGB)
        img_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_bgr
        
    except Exception as e:
        print(f"Error converting image {file_path}: {str(e)}")
        return None


def is_supported_image(file_path):
    """Check if the file is a supported image format."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


# ============================================================
# Configuration
# ============================================================
class Config:
    def __init__(self):
        self.SAVE_DIR = tempfile.mkdtemp()
        self.MODEL_PATH = "models/deeplab_mobilenetv3_best.pth"
        self.CLASSIFIER_MODEL_PATH = "models/rf_text_type_model.pkl"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_CLASSES = 2
        self.IMG_SIZE = 512
        self.SCALE_FACTOR = 0.95
        self.IS_SAVE_INTERMEDIATE = False
        self.SR_SCALE = 2
        self.APPLY_SR = True
        self.HOG_CONFIG = None

config = Config()

# Global storage for processed images (to ensure download works)
processed_images_storage = []

# ============================================================
# Model Loading (lazy loading for better startup)
# ============================================================
_models = {
    'segmentation': None,
    'classifier': None,
    'classifier_config': None,
    'enhancer': None
}

def get_segmentation_model():
    if _models['segmentation'] is None:
        from models.model_loader import load_segmentation_model
        _models['segmentation'] = load_segmentation_model()
    return _models['segmentation']

def get_classifier():
    if _models['classifier'] is None:
        from models.model_loader import load_classifier_model
        _models['classifier'], _models['classifier_config'] = load_classifier_model(config.CLASSIFIER_MODEL_PATH)
        config.HOG_CONFIG = _models['classifier_config']
    return _models['classifier']

def get_enhancer():
    if _models['enhancer'] is None:
        from filter_img_to_super_res import DocumentEnhancer
        _models['enhancer'] = DocumentEnhancer(apply_sr=config.APPLY_SR, sr_scale=config.SR_SCALE)
    return _models['enhancer']

# ============================================================
# Image Processing Functions
# ============================================================
def process_single_image(img_path, apply_sr=True, sr_scale=2):
    """Process a single image and return the corrected version."""
    from utils.image_processing import enhance_image, apply_perspective_correction, rotate_image, rotate_image2
    from utils.mask_processing import get_largest_component, fill_and_split_mask, get_regular_rectangular_mask
    from utils.orientation_detector import determine_orientation
    from utils.utils_ import predict_page_type
    
    # Update config
    config.APPLY_SR = apply_sr
    config.SR_SCALE = sr_scale
    
    img_name = os.path.basename(img_path)
    
    # Use our universal image loader
    img = convert_to_cv2_image(img_path)
    
    if img is None:
        return None, f"Error: Could not read image (format may not be supported)"
    
    # Get models
    model = get_segmentation_model()
    classifier_model = get_classifier()
    
    # Preprocessing
    enhanced = enhance_image(img)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (config.IMG_SIZE, config.IMG_SIZE)) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_resized - mean) / std
    tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(config.DEVICE)

    # Inference
    with torch.no_grad():
        out = model(tensor)["out"]
        mask = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Post-processing
    mask_rescaled = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    largest_region = get_largest_component(mask_rescaled)
    _, _, final_split_mask = fill_and_split_mask(largest_region)
    filtered_mask, box = get_regular_rectangular_mask(final_split_mask, config.SCALE_FACTOR)

    corrected_page = None
    status_msg = ""

    # Perspective Correction
    if box is not None:
        corrected_page = apply_perspective_correction(img, box)

        # Detect Orientation
        rotation_needed = determine_orientation(corrected_page)
        if rotation_needed != 0:
            corrected_page = rotate_image(corrected_page, rotation_needed)
            status_msg += f"Rotated by {rotation_needed}°. "

        # Predict page type (up/down)
        condition = predict_page_type(img_path, classifier_model)
        status_msg += f"Orientation: {condition}. "
        
        if condition == 'down':
            corrected_page = rotate_image2(corrected_page, 180)
            status_msg += "Flipped 180°. "

        # Super Resolution
        if apply_sr and sr_scale in [2, 4]:
            enhancer = get_enhancer()
            corrected_page = enhancer.enhance(corrected_page)
            status_msg += f"SR applied (scale: {sr_scale}x). "

        status_msg = f"✓ Processed successfully. {status_msg}"
    else:
        status_msg = "⚠ Could not detect document boundaries"
        corrected_page = img

    # Convert BGR to RGB for display
    if corrected_page is not None:
        corrected_page_rgb = cv2.cvtColor(corrected_page, cv2.COLOR_BGR2RGB)
        return corrected_page_rgb, status_msg
    
    return None, status_msg


def process_images(images, apply_sr, sr_scale, progress=gr.Progress()):
    """Process multiple images and return results."""
    global processed_images_storage
    processed_images_storage = []  # Clear previous results
    
    if images is None or len(images) == 0:
        return [], "No images uploaded"
    
    results = []
    status_messages = []
    
    for i, img_file in enumerate(progress.tqdm(images, desc="Processing images")):
        try:
            # Get the file path
            if hasattr(img_file, 'name'):
                img_path = img_file.name
            else:
                img_path = img_file
            
            filename = os.path.basename(img_path)
            
            # Check if format is supported
            if not is_supported_image(img_path):
                status_messages.append(f"{filename}: ⚠ Unsupported format")
                continue
            
            processed_img, status = process_single_image(img_path, apply_sr, sr_scale)
            
            if processed_img is not None:
                results.append((processed_img, filename))
                # Store for download
                processed_images_storage.append({
                    'image': processed_img,
                    'filename': filename
                })
                status_messages.append(f"{filename}: {status}")
            else:
                status_messages.append(f"{filename}: Failed - {status}")
                
        except Exception as e:
            fname = os.path.basename(img_path) if 'img_path' in dir() else 'Image'
            status_messages.append(f"{fname}: Error - {str(e)}")
    
    return results, "\n".join(status_messages)


def save_results():
    """Save processed images to a zip file for download."""
    global processed_images_storage
    
    if not processed_images_storage or len(processed_images_storage) == 0:
        print("No images to save")
        return None
    
    # Create temp directory for saving
    temp_dir = tempfile.mkdtemp()
    img_dir = os.path.join(temp_dir, "processed_documents")
    os.makedirs(img_dir, exist_ok=True)
    
    saved_count = 0
    
    for i, item in enumerate(processed_images_storage):
        try:
            img = item.get('image')
            filename = item.get('filename', f'processed_{i}.png')
            
            if img is None:
                continue
            
            # Create output filename (always save as PNG for compatibility)
            base, ext = os.path.splitext(filename)
            # Convert HEIC/HEIF/WebP output to PNG for better compatibility
            if ext.lower() in {'.heic', '.heif', '.avif', '.webp'}:
                ext = '.png'
            elif not ext:
                ext = '.png'
            
            save_filename = f"{base}_corrected{ext}"
            save_path = os.path.join(img_dir, save_filename)
            
            # Save based on type
            if isinstance(img, np.ndarray):
                img_pil = Image.fromarray(img.astype(np.uint8))
                img_pil.save(save_path, quality=95)
                saved_count += 1
            elif isinstance(img, Image.Image):
                img.save(save_path, quality=95)
                saved_count += 1
            elif isinstance(img, str) and os.path.exists(img):
                shutil.copy2(img, save_path)
                saved_count += 1
                
        except Exception as e:
            print(f"Error saving image {i}: {str(e)}")
            continue
    
    if saved_count == 0:
        print("No images were saved")
        return None
    
    # Create zip file
    zip_base = os.path.join(temp_dir, "processed_documents")
    zip_path = shutil.make_archive(zip_base, 'zip', img_dir)
    
    print(f"Created zip at: {zip_path} with {saved_count} images")
    return zip_path


def clear_all():
    """Clear all inputs and outputs."""
    global processed_images_storage
    processed_images_storage = []
    return None, [], "", None


def get_supported_formats_text():
    """Return a formatted string of supported formats."""
    formats = sorted(list(SUPPORTED_EXTENSIONS))
    return ", ".join(formats)


# ============================================================
# Custom CSS for Styling
# ============================================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* {
    font-family: 'Space Grotesk', sans-serif !important;
}

.gradio-container {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1f 100%) !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 1rem;
}

.main-header h1 {
    font-size: 3rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em;
}

.main-header p {
    color: #a1a1aa !important;
    font-size: 1.1rem !important;
    font-weight: 400;
}

.upload-section {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(10px);
}

.control-panel {
    background: rgba(124, 58, 237, 0.1) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6) !important;
}

.gr-button-secondary {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    color: #e4e4e7 !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.gr-button-secondary:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}

.gallery-container {
    background: rgba(0, 0, 0, 0.3) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.status-box {
    background: rgba(0, 212, 255, 0.05) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

.gr-form {
    background: transparent !important;
    border: none !important;
}

.gr-input, .gr-dropdown {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: #e4e4e7 !important;
}

.gr-checkbox {
    accent-color: #7c3aed !important;
}

label {
    color: #a1a1aa !important;
    font-weight: 500 !important;
}

.feature-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(124, 58, 237, 0.2));
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #00d4ff;
    margin: 0 4px;
}

.info-card {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-top: 1rem !important;
}

.info-card h3 {
    color: #f472b6 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
}

.info-card p {
    color: #71717a !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
}

.format-badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.7rem;
    color: #00d4ff;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace !important;
}

footer {
    display: none !important;
}
"""

# ============================================================
# Gradio Interface
# ============================================================
def create_interface():
    # Build format badges HTML
    format_badges = " ".join([f'<span class="format-badge">{ext}</span>' for ext in sorted(SUPPORTED_EXTENSIONS)])
    
    heic_status = "✓ HEIC/HEIF Enabled" if HEIC_SUPPORTED else "⚠ HEIC not available (install pillow-heif)"
    
    with gr.Blocks(css=custom_css, title="DocuVision Pro", theme=gr.themes.Base()) as demo:
        
        # Header
        gr.HTML(f"""
            <div class="main-header">
                <h1>📄 DocuVision Pro</h1>
                <p>AI-Powered Document Image Processing & Correction</p>
                <div style="margin-top: 1rem;">
                    <span class="feature-badge">🔲 Perspective Correction</span>
                    <span class="feature-badge">🔄 Auto Rotation</span>
                    <span class="feature-badge">✨ Super Resolution</span>
                    <span class="feature-badge">🎯 Text Detection</span>
                </div>
                <div style="margin-top: 0.75rem;">
                    <span class="feature-badge" style="background: rgba(34, 197, 94, 0.2); border-color: rgba(34, 197, 94, 0.3); color: #22c55e;">
                        {heic_status}
                    </span>
                </div>
            </div>
        """)
        
        with gr.Row():
            # Left Column - Upload & Controls
            with gr.Column(scale=1):
                gr.HTML('<div class="upload-section">')
                
                file_input = gr.File(
                    label="📁 Upload Document Images",
                    file_count="multiple",
                    file_types=["image"],
                    elem_classes=["upload-area"]
                )
                
                gr.HTML('</div>')
                
                gr.HTML('<div class="control-panel" style="margin-top: 1rem;">')
                
                with gr.Group():
                    gr.HTML('<p style="color: #a855f7; font-weight: 600; margin-bottom: 0.5rem;">⚙️ Processing Options</p>')
                    
                    apply_sr = gr.Checkbox(
                        label="Enable Super Resolution",
                        value=True,
                        info="Enhance image quality using AI upscaling"
                    )
                    
                    sr_scale = gr.Radio(
                        choices=[2, 4],
                        value=2,
                        label="SR Scale Factor",
                        info="2x for balance, 4x for maximum quality"
                    )
                
                gr.HTML('</div>')
                
                with gr.Row():
                    process_btn = gr.Button(
                        "🚀 Process Documents",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        size="lg"
                    )
                
                # Info Card with supported formats
                gr.HTML(f"""
                    <div class="info-card">
                        <h3>💡 Tips for Best Results</h3>
                        <p>
                            • Use well-lit images with clear document boundaries<br>
                            • Multiple files can be processed at once<br>
                            • Higher SR scale = better quality but slower processing
                        </p>
                        <h3 style="margin-top: 0.75rem;">📷 Supported Formats</h3>
                        <div style="margin-top: 0.5rem;">
                            {format_badges}
                        </div>
                    </div>
                """)
            
            # Right Column - Results
            with gr.Column(scale=2):
                gr.HTML('<div class="gallery-container">')
                
                gallery = gr.Gallery(
                    label="🖼️ Processed Results",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    object_fit="contain",
                    height="500px"
                )
                
                gr.HTML('</div>')
                
                gr.HTML('<div class="status-box" style="margin-top: 1rem;">')
                
                status_output = gr.Textbox(
                    label="📋 Processing Status",
                    lines=6,
                    max_lines=10,
                    interactive=False,
                    placeholder="Processing status will appear here..."
                )
                
                gr.HTML('</div>')
                
                download_btn = gr.Button(
                    "📥 Download All Results",
                    variant="secondary",
                    size="lg"
                )
                
                download_file = gr.File(
                    label="📦 Download Ready",
                    visible=True,
                    interactive=False
                )
        
        # Event Handlers
        process_btn.click(
            fn=process_images,
            inputs=[file_input, apply_sr, sr_scale],
            outputs=[gallery, status_output]
        )
        
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[file_input, gallery, status_output, download_file]
        )
        
        # Download button - no inputs needed, uses global storage
        download_btn.click(
            fn=save_results,
            inputs=[],
            outputs=[download_file]
        )
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; padding: 2rem 0; color: #52525b; font-size: 0.85rem;">
                <p>DocuVision Pro • Powered by Deep Learning</p>
                <p style="font-size: 0.75rem; margin-top: 0.5rem;">
                    Using DeepLab for segmentation • Random Forest for orientation • Real-ESRGAN for super-resolution
                </p>
            </div>
        """)
    
    return demo


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    print(f"Supported image formats: {get_supported_formats_text()}")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )