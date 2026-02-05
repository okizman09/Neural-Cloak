"""White-box adversarial attacks targeting face embeddings with feature masking.

This module provides PGD-style attacks that maximize embedding distance
while confining perturbations to specific facial features (Eyes, Nose, Mouth)
to preserve overall visual quality.
"""
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
import cv2

# Import from local vendored library or installed package
try:
    import torch
    import torch.nn.functional as F
    from facenet_pytorch import InceptionResnetV1
    _HAS_DEPS = True
    _IMPORT_ERROR = None
except ImportError as e:
    print(f"DEBUG: Import Error in adversary: {e}")
    _IMPORT_ERROR = str(e)
    torch = None
    F = None
    InceptionResnetV1 = None
    _HAS_DEPS = False

def get_import_error():
    return _IMPORT_ERROR

def has_deps() -> bool:
    return _HAS_DEPS

def has_cuda() -> bool:
    if not _HAS_DEPS:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

def load_embedding_model(device: str = 'cpu'):
    if not _HAS_DEPS:
        raise RuntimeError('Deep learning dependencies not installed.')
    # Pretrained on vggface2
    model = InceptionResnetV1(pretrained='vggface2').eval()
    model = model.to(device)
    return model

def _pil_to_torch(img: Image.Image, device: str):
    # Convert PIL RGB to tensor [-1, 1]
    # Ensure RGB
    img = img.convert('RGB')
    arr = np.array(img).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = (tensor / 127.5) - 1.0
    return tensor

def _torch_to_pil(tensor):
    t = tensor.detach().cpu().squeeze(0)
    arr = ((t + 1.0) * 127.5).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr)

def create_feature_mask(pil_img: Image.Image, boxes: list) -> "torch.Tensor":
    """
    Creates a soft binary mask (1.0 on features, 0.0 elsewhere).
    Falls back to Oval mask on faces if no landmarks available.
    """
    if not _HAS_DEPS:
        raise RuntimeError("Missing dependencies")
        
    mask_img = Image.new('L', pil_img.size, 0)
    draw = ImageDraw.Draw(mask_img)
    
    # If boxes are empty, try to detect
    if not boxes:
         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
         # Convert PIL to CV2 grayscale
         arr_gray = np.array(pil_img.convert('L'))
         detected = face_cascade.detectMultiScale(arr_gray, 1.1, 4)
         boxes = [(x, y, w, h) for (x, y, w, h) in detected]

    for (x, y, w, h) in boxes:
        # Draw soft oval for the face features
        # Widen to 100% of box to ensure we hit all features
        cx, cy = x + w//2, y + h//2
        nw, nh = int(w * 1.0), int(h * 1.0)
        nx, ny = cx - nw//2, cy - nh//2
        draw.ellipse([nx, ny, nx+nw, ny+nh], fill=255)
            
    # Convert to torch [1, 3, H, W]
    mask_arr = np.array(mask_img).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    return mask_tensor

def pgd_attack_targeted(
    model, 
    orig_img: Image.Image, 
    eps: float = 8.0, 
    alpha: float = 2.0, 
    steps: int = 10, 
    device: str = 'cpu',
    boxes: list = None
):
    """
    Applied PGD attack constrained to facial features.
    
    OPTIMIZATION:
    Instead of processing the full image (which can be 4K+), we:
    1. Iterate through each detected face.
    2. Crop the face with some padding.
    3. Run PGD on the small crop.
    4. Paste the adversarial crop back.
    """
    if not _HAS_DEPS:
        raise RuntimeError('Dependencies missing')

    device = torch.device(device)
    model = model.to(device)
    
    # 1. Detect faces if not provided
    if not boxes:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = np.array(orig_img.convert('L'))
        detected = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(detected) > 0:
            boxes = [(x, y, w, h) for (x, y, w, h) in detected]
        else:
            boxes = []
    
    if len(boxes) == 0:
        print("DEBUG: No faces found for PGD.")
        return orig_img, 0.0

    # We will build the final image by pasting modified crops onto the original
    final_img = orig_img.copy()
    
    total_dist = 0.0
    processed_count = 0
    
    print(f"DEBUG: Starting Crop-Based PGD. Eps={eps}, Alpha={alpha}, Steps={steps}, Faces={len(boxes)}")
    
    W_full, H_full = orig_img.size
    
    for i, (x, y, w, h) in enumerate(boxes):
        # --- 1. Define Crop Region (add 20% padding) ---
        pad_w = int(w * 0.2)
        pad_h = int(h * 0.2)
        
        # Coordinates of the ROI in full image
        cx1 = max(0, x - pad_w)
        cy1 = max(0, y - pad_h)
        cx2 = min(W_full, x + w + pad_w)
        cy2 = min(H_full, y + h + pad_h)
        
        # The cropped PIL image
        crop = orig_img.crop((cx1, cy1, cx2, cy2))
        
        # Face coordinates relative to the crop
        rx = x - cx1
        ry = y - cy1
        # rw, rh are just w, h
        
        # --- 2. Prepare PGD for this crop ---
        
        # Convert crop to tensor
        x_crop = _pil_to_torch(crop, device)
        
        # Create mask for this single face relative to crop
        # create_feature_mask expects a list of boxes
        mask = create_feature_mask(crop, [(rx, ry, w, h)]).to(device)
        
        # Make working copy
        x_adv = x_crop.clone().detach()
        x_adv.requires_grad = True
        
        eps_scaled = eps / 127.5
        alpha_scaled = alpha / 127.5
        
        # Compute original embedding target for this face
        # Extract face from crop tensor
        def get_face_tensor(tensor_img, rect):
            bx, by, bw, bh = rect
            return tensor_img[:, :, by:by+bh, bx:bx+bw]
            
        orig_face_t = get_face_tensor(x_crop, (rx, ry, w, h))
        if orig_face_t.shape[2] == 0 or orig_face_t.shape[3] == 0:
            print(f"DEBUG: Skipped face {i} due to empty crop.")
            continue
            
        # Target embedding
        with torch.no_grad():
             orig_resized = F.interpolate(orig_face_t, size=(160, 160), mode='bilinear', align_corners=False)
             target_emb = model(orig_resized).detach()

        # --- 3. Run Optimization Loop ---
        final_loss = 0.0
        
        for step in range(steps):
             if x_adv.grad is not None:
                 x_adv.grad.zero_()
             
             # Get adversarial face
             adv_face_t = get_face_tensor(x_adv, (rx, ry, w, h))
             adv_resized = F.interpolate(adv_face_t, size=(160, 160), mode='bilinear', align_corners=False)
             adv_emb = model(adv_resized)
             
             # Loss (Minimize Cosine Similarity)
             sim = F.cosine_similarity(adv_emb, target_emb)
             loss = sim.mean()
             
             loss.backward()
             
             # Update with mask
             grad = x_adv.grad.data
             grad_masked = grad * mask
             
             x_adv.data = x_adv.data - alpha_scaled * grad_masked.sign()
             
             # Projection
             delta = torch.clamp(x_adv.data - x_crop, -eps_scaled, eps_scaled)
             x_adv.data = torch.clamp(x_crop + delta, -1.0, 1.0)
             
             final_loss = loss.item()
        
        # --- 4. Finalize Crop ---
        
        # Compute final distance for reporting
        with torch.no_grad():
             final_face_t = get_face_tensor(x_adv, (rx, ry, w, h))
             final_resized = F.interpolate(final_face_t, size=(160, 160), mode='bilinear', align_corners=False)
             final_emb_check = model(final_resized)
             dist = torch.norm(final_emb_check - target_emb).item()
             total_dist += dist
             processed_count += 1
        
        # Convert back to PIL
        adv_crop_pil = _torch_to_pil(x_adv)
        
        # Paste back
        final_img.paste(adv_crop_pil, (cx1, cy1))
        
        # print(f"DEBUG: Face {i} done. Dist={dist:.2f}")

    avg_dist = total_dist / processed_count if processed_count > 0 else 0.0
    print(f"DEBUG: Final Avg Dist={avg_dist:.3f}")
    
    return final_img, avg_dist
# Alias for compatibility if needed
lite_adversarial_attack = pgd_attack_targeted
pgd_attack_embedding = pgd_attack_targeted
has_torch = has_deps
