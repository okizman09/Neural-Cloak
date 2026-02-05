import io
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))

def cloak_image(pil_img: Image.Image, strength: float = 4.0, seed: int | None = None) -> Image.Image:
    """Apply a Gaussian-like adversarial perturbation to an image.
    Used as a fallback if PGD dependencies fail.
    
    Args:
        pil_img: PIL.Image input
        strength: noise strength (roughly 0-20).
        seed: optional RNG seed

    Returns:
        cloaked PIL.Image
    """
    if seed is not None:
        np.random.seed(seed)

    arr = pil_to_np(pil_img).astype(np.float32)

    # noise scale relative to image dynamic range
    sigma = max(0.5, float(strength) * 0.8)

    noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)

    cloaked = arr + noise
    cloaked = np.clip(cloaked, 0, 255)

    return np_to_pil(cloaked)


def compute_ssim(pil_a: Image.Image, pil_b: Image.Image) -> float:
    # Convert to 8-bit grayscale arrays so skimage's SSIM works without
    # ambiguity about data range.
    a = np.array(pil_a.convert("L"), dtype=np.uint8)
    b = np.array(pil_b.convert("L"), dtype=np.uint8)
    # specify data_range explicitly to avoid ValueError when inputs are
    # floating point or when skimage cannot infer the range.
    s, _ = compare_ssim(a, b, full=True, data_range=255)
    return float(s)


def pil_save_with_exif(pil_img: Image.Image, exif_bytes: bytes | None = None, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    if exif_bytes:
        pil_img.save(buf, format="JPEG", exif=exif_bytes, quality=quality, subsampling=0)
    else:
        pil_img.save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    return buf.read()
