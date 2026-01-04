import io
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
import cv2


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def cloak_image(pil_img: Image.Image, strength: float = 4.0, seed: int | None = None) -> Image.Image:
    """Apply a Gaussian-like adversarial perturbation to an image.

    Args:
        pil_img: PIL.Image input
        strength: noise strength (roughly 0-20). Higher = stronger perturbation.
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


def pil_save_with_exif(pil_img: Image.Image, exif_bytes: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    if exif_bytes:
        pil_img.save(buf, format="JPEG", exif=exif_bytes)
    else:
        pil_img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def find_max_strength_for_ssim(pil_img: Image.Image, target_ssim: float = 0.95, max_strength: float = 20.0, tol: float = 0.25, seed: int | None = None):
    """Find the largest cloak strength that still satisfies SSIM >= target_ssim.

    Uses a binary-search style approach on strength in [0, max_strength].

    Returns a tuple: (best_strength, cloaked_image, ssim_value)
    """
    # Handle trivial case
    if target_ssim >= 1.0:
        return 0.0, pil_img, 1.0

    low = 0.0
    high = float(max_strength)
    best_strength = 0.0
    best_cloak = pil_img
    best_ssim = compute_ssim(pil_img, pil_img)

    # Binary-search to find the maximum strength that keeps SSIM >= target
    while (high - low) > tol:
        mid = (low + high) / 2.0
        candidate = cloak_image(pil_img, strength=mid, seed=seed)
        s = compute_ssim(pil_img, candidate)

        if s >= target_ssim:
            # mid is acceptable; try stronger noise
            best_strength = mid
            best_cloak = candidate
            best_ssim = s
            low = mid
        else:
            # too strong, reduce
            high = mid

    return best_strength, best_cloak, best_ssim


def find_min_strength_for_disruption(pil_img: Image.Image, get_detections_fn, boxes: list[tuple] | None = None, min_ssim: float = 0.85, min_embed_dist: float = 0.6, max_strength: float = 30.0, max_attempts: int = 20, seed: int | None = None):
    """Binary-search for the MINIMUM strength needed to disrupt all detections AND embeddings.

    Args:
        pil_img: PIL.Image input
        get_detections_fn: callable(cloaked_image) -> (num_detections, avg_embed_dist or None)
                          Returns (detection_count, embedding_distance). If embeddings unavailable, pass None for distance.
        boxes: optional list of (x, y, w, h) face boxes. If provided, applies region-targeted cloaking (more efficient).
        min_ssim: minimum acceptable SSIM during search
        min_embed_dist: minimum embedding L2 distance required (if embeddings available)
        max_strength: maximum strength to search up to
        max_attempts: max binary-search iterations
        seed: RNG seed

    Returns:
        (min_strength, cloaked_image, num_detections, avg_embed_dist, final_ssim)
    """
    if seed is not None:
        np.random.seed(seed)

    low = 0.0
    high = float(max_strength)
    best_strength = None
    best_cloak = None
    best_detections = float('inf')
    best_embed_dist = 0.0
    best_ssim = 1.0

    for attempt in range(max_attempts):
        mid = (low + high) / 2.0
        # If boxes provided, use region-targeted cloaking (1.2x sigma for stronger effect)
        if boxes is not None and len(boxes) > 0:
            candidate = cloak_face_regions(pil_img, boxes, face_strength=mid, seed=seed, blur_radius=31)
        else:
            candidate = cloak_image(pil_img, strength=mid, seed=seed)
        ssim = compute_ssim(pil_img, candidate)

        # check disruption
        detections, embed_dist = get_detections_fn(candidate)

        # criteria: detections reduced to 0 AND (embeddings far OR no embedding support)
        disrupted = (detections == 0) and (embed_dist is None or embed_dist >= min_embed_dist)

        if disrupted and ssim >= min_ssim:
            # success: found acceptable strength, try lower
            best_strength = mid
            best_cloak = candidate
            best_detections = detections
            best_embed_dist = embed_dist if embed_dist is not None else 0.0
            best_ssim = ssim
            high = mid
        else:
            # not disrupted enough or SSIM too low; go stronger
            low = mid

        if (high - low) < 0.1:
            break

    if best_strength is None:
        # could not find acceptable strength; return strongest attempt
        best_strength = max_strength
        if boxes is not None and len(boxes) > 0:
            best_cloak = cloak_face_regions(pil_img, boxes, face_strength=best_strength, seed=seed, blur_radius=31)
        else:
            best_cloak = cloak_image(pil_img, strength=best_strength, seed=seed)
        best_ssim = compute_ssim(pil_img, best_cloak)
        detections, embed_dist = get_detections_fn(best_cloak)
        best_detections = detections
        best_embed_dist = embed_dist if embed_dist is not None else 0.0

    return best_strength, best_cloak, best_detections, best_embed_dist, best_ssim



def cloak_face_regions(pil_img: Image.Image, boxes: list[tuple], face_strength: float = 12.0, seed: int | None = None, blur_radius: int = 31) -> Image.Image:
    """Apply stronger noise only inside provided face bounding boxes.

    Args:
        pil_img: input PIL image
        boxes: list of (x, y, w, h) tuples in image coordinates
        face_strength: noise strength to apply inside face regions
        seed: RNG seed
        blur_radius: kernel size for smoothing mask (should be odd)

    Returns:
        PIL.Image with region-applied noise
    """
    if seed is not None:
        np.random.seed(seed)

    arr = pil_to_np(pil_img).astype(np.float32)
    h, w = arr.shape[:2]

    # If no boxes, fallback to global cloak
    if not boxes:
        return cloak_image(pil_img, strength=face_strength, seed=seed)

    # Work on uint8 RGB arrays for OpenCV seamlessClone (which expects BGR)
    dst = arr.astype(np.uint8).copy()

    # For each face box, create a noisy source patch and seamlessly clone it back
    for (x, y, bw, bh) in boxes:
        # modest expansion to cover hair/edges
        expand_x = int(bw * 0.10)
        expand_y = int(bh * 0.10)
        x1 = max(0, x - expand_x)
        y1 = max(0, y - expand_y)
        x2 = min(w, x + bw + expand_x)
        y2 = min(h, y + bh + expand_y)

        if x2 <= x1 or y2 <= y1:
            continue

        src_patch = dst[y1:y2, x1:x2].astype(np.float32)

        # face patch noise
        sigma_face = max(0.5, float(face_strength) * 1.2)
        noise_patch = np.random.normal(loc=0.0, scale=sigma_face, size=src_patch.shape).astype(np.float32)
        src_noisy = np.clip(src_patch + noise_patch, 0, 255).astype(np.uint8)

        # optional small blur on source to avoid sharp boundaries
        src_noisy = cv2.GaussianBlur(src_noisy, (3, 3), 0)

        # Apply a mild elastic warp to distort facial geometry (reduces detector/embedding reliability)
        ph, pw = src_noisy.shape[:2]
        # displacement strength scaled to patch size and face_strength
        alpha = max(1.0, float(face_strength) * 0.15)
        # generate random displacement fields and smooth them
        dx = (np.random.rand(ph, pw).astype(np.float32) * 2 - 1)
        dy = (np.random.rand(ph, pw).astype(np.float32) * 2 - 1)
        # smooth with Gaussian to create coherent flows
        ksize = max(3, int(min(ph, pw) / 6))
        if ksize % 2 == 0:
            ksize += 1
        dx = cv2.GaussianBlur(dx, (ksize, ksize), 0) * alpha
        dy = cv2.GaussianBlur(dy, (ksize, ksize), 0) * alpha

        # build remap coordinates
        grid_x, grid_y = np.meshgrid(np.arange(pw), np.arange(ph))
        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)
        try:
            warped = cv2.remap(src_noisy, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        except Exception:
            warped = src_noisy
        src_noisy = warped

        # create mask for the patch (full white) and feather it
        mask_patch = np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255
        # feather mask edges with a small kernel
        k = int(max(3, blur_radius // 8))
        if k % 2 == 0:
            k += 1
        mask_patch = cv2.GaussianBlur(mask_patch, (k, k), 0)

        # seamlessClone expects BGR images
        src_bgr = cv2.cvtColor(src_noisy, cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

        center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

        try:
            cloned = cv2.seamlessClone(src_bgr, dst_bgr, mask_patch, center, cv2.NORMAL_CLONE)
            # update destination (convert back to RGB)
            dst = cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB)
        except Exception:
            # fallback: paste noisy patch with alpha blending
            alpha = (mask_patch.astype(np.float32) / 255.0)[:, :, None]
            dst[y1:y2, x1:x2] = (src_noisy.astype(np.float32) * alpha + dst[y1:y2, x1:x2].astype(np.float32) * (1 - alpha)).astype(np.uint8)

    # Add a tiny amount of global noise to reduce any residual seams
    sigma_global = max(0.15, float(face_strength) * 0.06)
    global_noise = np.random.normal(loc=0.0, scale=sigma_global, size=dst.shape).astype(np.float32)
    dst = np.clip(dst.astype(np.float32) + global_noise, 0, 255).astype(np.uint8)

    return np_to_pil(dst)


def aggressive_disrupt(pil_img: Image.Image, strength: float = 20.0, seed: int | None = None) -> Image.Image:
    """Apply EXTREME multi-layer disruption to defeat face detection and AI vision models.
    
    Combines:
    1. Extreme high-frequency noise (defeats feature extraction)
    2. Aggressive color space corruption (defeats color-based analysis)
    3. Heavy randomized block corruption (defeats pattern recognition)
    4. Frequency domain disruption (defeats DCT/FFT-based analysis)
    5. Random pixel scrambling in patches
    
    Args:
        pil_img: Input PIL image
        strength: Disruption strength (10-50 recommended; >40 = heavily corrupted)
        seed: RNG seed
    
    Returns:
        Extremely corrupted PIL image
    """
    if seed is not None:
        np.random.seed(seed)
    
    arr = pil_to_np(pil_img).astype(np.float32)
    h, w = arr.shape[:2]
    
    # Layer 1: EXTREME high-frequency noise (much stronger than before)
    hf_noise = np.random.normal(loc=0.0, scale=strength * 1.2, size=arr.shape).astype(np.float32)
    arr = arr + hf_noise
    
    # Layer 2: Extreme color channel randomization (larger shifts)
    for c in range(3):
        channel_shift = np.random.randint(-100, 100)  # Doubled from ±50
        arr[:, :, c] = np.clip(arr[:, :, c] + channel_shift, 0, 255)
    
    # Layer 3: AGGRESSIVE random block corruption (larger blocks, more coverage)
    block_size = max(20, int(min(h, w) / 15))  # Larger blocks
    num_blocks = int((strength / 20.0) * 80)  # ~4x more blocks
    for _ in range(num_blocks):
        y = np.random.randint(0, max(1, h - block_size))
        x = np.random.randint(0, max(1, w - block_size))
        # Extreme noise in block
        block_noise = np.random.normal(loc=0.0, scale=150.0, size=(block_size, block_size, 3))
        arr[y:y+block_size, x:x+block_size] = np.clip(
            arr[y:y+block_size, x:x+block_size] + block_noise, 0, 255
        )
    
    # Layer 4: Pixel-level scrambling in random patches (defeats spatial correlation)
    scramble_patches = int((strength / 20.0) * 30)
    for _ in range(scramble_patches):
        py = np.random.randint(0, max(1, h - 50))
        px = np.random.randint(0, max(1, w - 50))
        patch = arr[py:py+50, px:px+50].copy()
        # Randomly scramble pixels in patch
        patch_flat = patch.reshape(-1, 3)
        np.random.shuffle(patch_flat)
        arr[py:py+50, px:px+50] = patch_flat.reshape(50, 50, 3)
    
    # Layer 5: Frequency-domain disruption (corrupt DCT coefficients)
    # Process in smaller tiles to avoid memory issues
    tile_size = 32
    for ty in range(0, h, tile_size):
        for tx in range(0, w, tile_size):
            th = min(tile_size, h - ty)
            tw = min(tile_size, w - tx)
            for c in range(3):
                tile = arr[ty:ty+th, tx:tx+tw, c].copy()
                if tile.shape[0] > 1 and tile.shape[1] > 1:
                    # Apply random phase shifts to disrupt any frequency patterns
                    if np.random.random() > 0.7:  # 30% chance per tile
                        tile = np.roll(tile, np.random.randint(-3, 4), axis=0)
                        tile = np.roll(tile, np.random.randint(-3, 4), axis=1)
                        arr[ty:ty+th, tx:tx+tw, c] = tile
    
    # Layer 6: Cross-channel mixing (defeats RGB analysis)
    for _ in range(int(strength / 10.0)):
        y1, y2 = np.random.randint(0, h, 2)
        x1, x2 = np.random.randint(0, w, 2)
        if y1 > y2: y1, y2 = y2, y1
        if x1 > x2: x1, x2 = x2, x1
        # Swap channels in random regions
        c1, c2 = np.random.choice(3, 2, replace=False)
        arr[y1:y2, x1:x2, c1], arr[y1:y2, x1:x2, c2] = \
            arr[y1:y2, x1:x2, c2].copy(), arr[y1:y2, x1:x2, c1].copy()
    
    # Final clipping and return
    arr = np.clip(arr, 0, 255)
    return np_to_pil(arr)
