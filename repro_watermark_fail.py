import io
from PIL import Image
import numpy as np
from watermark import embed_watermark, extract_watermark, generate_watermark_id
from cloak import pil_save_with_exif

def test_watermark_repro():
    # Create a smooth gradient image (simulate natural photo)
    print("1. Creating gradient dummy image...")
    w, h = 512, 512
    x = np.linspace(0, 255, w)
    y = np.linspace(0, 255, h)
    xv, yv = np.meshgrid(x, y)
    img_array = np.stack([xv, yv, np.zeros_like(xv)], axis=-1).astype(np.uint8)
    # Add slight noise to make it realistic but not white noise
    noise = np.random.normal(0, 5, (h, w, 3))
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    original_img = Image.fromarray(img_array)
    
    print("2. Embedding Watermark...")
    wm_id = generate_watermark_id()
    print(f"   ID to embed: {wm_id}")
    watermarked_img = embed_watermark(original_img, wm_id)
    
    print("3. Saving as JPEG (Quality=95)...")
    # Simulate the app's save function
    jpeg_bytes = pil_save_with_exif(watermarked_img, quality=95)
    
    print(f"   Saved {len(jpeg_bytes)} bytes.")
    
    print("4. Reloading from bytes (Simulating Upload)...")
    uploaded_img = Image.open(io.BytesIO(jpeg_bytes))
    
    print("5. Extracting...")
    extracted_id = extract_watermark(uploaded_img)
    
    print(f"   Extracted ID: '{extracted_id}'")
    
    if extracted_id == wm_id:
        print("SUCCESS: Watermark survived.")
    else:
        print("FAILURE: Watermark lost or corrupted.")
        # Debug: check if it works without JPEG compression
        print("   DEBUG: Checking pre-save image...")
        direct_extract = extract_watermark(watermarked_img)
        print(f"   Direct Extraction: '{direct_extract}'")

if __name__ == "__main__":
    test_watermark_repro()
