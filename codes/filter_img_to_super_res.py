import os
import cv2
import numpy as np
from pathlib import Path
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet 

class DocumentEnhancer:
    def __init__(self, apply_sr: bool = False, sr_scale: int = 2, device: str = 'cpu'):
        """
        Initializes the enhancer. If apply_sr is True, the Real-ESRGAN model 
        is loaded into memory once.
        """
        self.apply_sr = apply_sr
        self.sr_scale = sr_scale
        self.device = device
        self.upsampler = None

        if self.apply_sr:
            self._init_upsampler()

    def _init_upsampler(self):
        """Internal method to load the model into memory."""
        if self.sr_scale not in [2, 4]:
            raise ValueError("Only scale=2 or 4 is supported")

        model_urls = {
            2: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            4: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        }

        print(f"⚡ Initializing Real-ESRGAN x{self.sr_scale} on {self.device} (Once)...")
        
        # Define architecture
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                        num_block=23, num_grow_ch=32, scale=self.sr_scale)

        # Initialize the Upsampler
        self.upsampler = RealESRGANer(
            scale=self.sr_scale,
            model_path=model_urls[self.sr_scale],
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=self.device
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Full enhancement pipeline: Contrast -> Denoise -> Sharpen -> SR
        """
        # 1️⃣ CLAHE contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2️⃣ Edge-preserving denoising
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        # 3️⃣ Sharpening (Unsharp Mask)
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
        sharpened = cv2.addWeighted(enhanced, 1.6, blur, -0.6, 0)

        # 4️⃣ Text edge enhancement
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 10)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        final = cv2.addWeighted(sharpened, 0.85, edges, 0.15, 0)

        # 5️⃣ Optional Super Resolution
        if self.apply_sr and self.upsampler is not None:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
            # Enhance
            output, _ = self.upsampler.enhance(img_rgb, outscale=self.sr_scale)
            # Convert back to BGR
            final = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return final

def process_and_save(input_path: str, output_dir: str, apply_sr: bool = True, sr_scale: int = 2):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate the class ONCE
    enhancer = DocumentEnhancer(apply_sr=apply_sr, sr_scale=sr_scale)

    def process_file(img_path: Path):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠ Warning: failed to read {img_path}, skipping")
            return
        
        # Use the persistent enhancer instance
        out = enhancer.enhance(img)
        
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), out)
        print(f"✅ Saved: {out_path}")

    # Execution logic
    if input_path.is_file():
        process_file(input_path)
    elif input_path.is_dir():
        imgs = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')]
        if not imgs:
            print(f"⚠ No image files found in {input_path}")
        else:
            for p in imgs:
                process_file(p)
    else:
        print(f"❌ Input path does not exist: {input_path}")

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    INPUT = "/mnt/storage1/workspace/arobin/page_orientation/test_img/photo.jpg"
    OUTPUT = "./enhanced_output"
    
    # This will now only load the model once for the entire batch
    process_and_save(INPUT, OUTPUT, apply_sr=True, sr_scale=2)