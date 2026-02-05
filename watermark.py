import cv2
import numpy as np
from PIL import Image
import uuid
import time
import struct

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR format."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL format."""
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def text_to_bits(text: str) -> str:
    """Convert string to a binary string."""
    bits = bin(int.from_bytes(text.encode('utf-8', 'surrogatepass'), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def bits_to_text(bits: str) -> str:
    """Convert binary string to text by processing 8-bit chunks."""
    chars = []
    # Process 8 bits at a time
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        # Only process complete bytes
        if len(byte) == 8:
            try:
                # Convert 8 bits to int, then to char
                chars.append(chr(int(byte, 2)))
            except:
                chars.append('?') # Placeholder for invalid utf-8 if using chr on raw bytes? 
                # actually chr() handles unicode code points, but we encoded utf-8 bytes.
                # So we should rebuild the bytes then decode.
    
    # Rebuild bytes
    try:
        # Convert list of chars (which are actually byte values 0-255 cast to chr) back to bytes
        byte_array = bytes([ord(c) for c in chars])
        return byte_array.decode('utf-8', 'ignore')
    except:
        return ""

def embed_watermark(pil_img: Image.Image, text: str) -> Image.Image:
    """
    Embeds an invisible watermark into the image using DCT (Discrete Cosine Transform).
    Robust against JPEG compression and minor resizing.
    
    Args:
        pil_img: Input PIL Image
        text: Text to embed (e.g., "ID:1234")
        
    Returns:
        Watermarked PIL Image
    """
    # 1. Prepare Image: Working in YCrCb color space, using the Y (Luma) channel is common,
    # but the Blue channel (in RGB) or Cr/Cb is often less perceptually sensitive.
    # Let's use the Blue channel of BGR for simplicity and robustness.
    img_bgr = pil_to_cv2(pil_img)
    h, w, _ = img_bgr.shape
    
    # 2. Prepare Message: Add start/end markers for reliability
    full_message = f"<START>{text}<END>"
    bits = text_to_bits(full_message)
    
    # Redundancy factor
    redundancy = 7
    
    # We will embed 1 bit per 8x8 block.
    # Capacity check
    needed_blocks = len(bits) * redundancy
    available_blocks = (h // 8) * (w // 8)
    
    if needed_blocks > available_blocks:
        print(f"Warning: Image too small for watermark. Needed {needed_blocks} blocks, have {available_blocks}.")
        return pil_img # Return original if too small
        
    # 3. Process 8x8 blocks
    # We embed in the Green channel (index 1) to avoid chroma subsampling issues
    b_channel = img_bgr[:, :, 1].astype(np.float32)
    
    bit_idx = 0
    repeat_idx = 0
    
    # Zigzag scan locations for mid-frequency coefficients (robust to compression)
    u, v = 1, 2 # Lower frequency for better robustness
    
    for r in range(0, h - 7, 8):
        for c in range(0, w - 7, 8):
            if bit_idx >= len(bits):
                break
                
            block = b_channel[r:r+8, c:c+8]
            dct_block = cv2.dct(block)
            
            # Embedding logic:
            # If bit is 1, ensure coefficient A > coefficient B + gap
            # If bit is 0, ensure coefficient B > coefficient A + gap
            # We modify two mid-freq coefficients.
            coeff_a = dct_block[u, v]
            coeff_b = dct_block[v, u]
            
            gap = 50.0 # Strength of watermark. Optimized for uint8 capacity vs robustness.
            
            current_bit = int(bits[bit_idx])
            
            if current_bit == 1:
                if coeff_a <= coeff_b + gap:
                    diff = (coeff_b + gap) - coeff_a
                    dct_block[u, v] += (diff / 2.0) + 1.0
                    dct_block[v, u] -= (diff / 2.0) + 1.0
            else: # bit == 0
                if coeff_b <= coeff_a + gap:
                    diff = (coeff_a + gap) - coeff_b
                    dct_block[v, u] += (diff / 2.0) + 1.0
                    dct_block[u, v] -= (diff / 2.0) + 1.0
            
            # Inverse DCT
            idct_block = cv2.idct(dct_block)
            b_channel[r:r+8, c:c+8] = idct_block
            
            # Move to next repetition or next bit
            repeat_idx += 1
            if repeat_idx >= redundancy:
                repeat_idx = 0
                bit_idx += 1
            
    # 4. Merge back
    img_bgr[:, :, 1] = np.clip(b_channel, 0, 255).astype(np.uint8)
    
    return cv2_to_pil(img_bgr)

def extract_watermark(pil_img: Image.Image) -> str:
    """
    Extracts the invisible watermark from the image.
    
    Args:
        pil_img: Input PIL Image
        
    Returns:
        Extracted text or empty string if failed.
    """
    img_bgr = pil_to_cv2(pil_img)
    h, w, _ = img_bgr.shape
    
    b_channel = img_bgr[:, :, 1].astype(np.float32)
    
    bits = ""
    u, v = 1, 2
    
    # Redundancy
    redundancy = 7
    current_votes = 0
    vote_count = 0
    
    for r in range(0, h - 7, 8):
        for c in range(0, w - 7, 8):
            block = b_channel[r:r+8, c:c+8]
            dct_block = cv2.dct(block)
            
            coeff_a = dct_block[u, v]
            coeff_b = dct_block[v, u]
            
            if coeff_a > coeff_b:
                current_votes += 1
            else:
                current_votes += 0 # explicit
            
            vote_count += 1
            
            if vote_count == redundancy:
                # Majority vote
                if current_votes > (redundancy / 2):
                    bits += "1"
                else:
                    bits += "0"
                
                # Reset
                vote_count = 0
                current_votes = 0
            
    # Try to find the message
    raw_text = bits_to_text(bits)
    
    # Look for markers
    start_marker = "<START>"
    end_marker = "<END>"
    
    start_idx = raw_text.find(start_marker)
    if start_idx != -1:
        end_idx = raw_text.find(end_marker, start_idx)
        if end_idx != -1:
            return raw_text[start_idx + len(start_marker) : end_idx]
            
    return ""

def generate_watermark_id() -> str:
    """Generates a concise unique ID string."""
    short_id = str(uuid.uuid4())[:8]
    ts = int(time.time())
    return f"NCP-{short_id}-{ts}"
