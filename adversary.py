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
except ImportError as e:
    print(f"DEBUG: Import Error in adversary: {e}")
    torch = None
    F = None
    InceptionResnetV1 = None
    _HAS_DEPS = False

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
    """
    if not _HAS_DEPS:
        raise RuntimeError('Dependencies missing')

    device = torch.device(device)
    model = model.to(device)
    
    # Prepare data
    x = _pil_to_torch(orig_img, device)
    
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

    # Create mask using the decided boxes
    mask = create_feature_mask(orig_img, boxes).to(device)
    
    # Debug mask coverage
    mask_pixels = mask[0,0,:,:].sum().item()
    total_pixels = x.shape[2] * x.shape[3]
    print(f"DEBUG: Mask covers {mask_pixels:.0f} pixels ({mask_pixels/total_pixels*100:.1f}%)")

    # If mask is empty (no faces), fallback to full image or return
    if mask.sum() == 0:
        return orig_img, 0.0

    # PGD Setup
    x_adv = x.clone().detach()
    x_adv.requires_grad = True
    
    eps_scaled = eps / 127.5
    alpha_scaled = alpha / 127.5
    
    # 2. Precompute original embeddings
    orig_crops = []
    
    # Resize transform for FaceNet (160x160)
    def get_crop(tensor_img, bx):
        x_c, y_c, w_c, h_c = bx
        return tensor_img[:, :, y_c:y_c+h_c, x_c:x_c+w_c]

    for (x_b, y_b, w_b, h_b) in boxes:
        crop = get_crop(x, (x_b, y_b, w_b, h_b))
        if crop.shape[2] == 0 or crop.shape[3] == 0: continue
        resized = F.interpolate(crop, size=(160, 160), mode='bilinear', align_corners=False)
        with torch.no_grad():
            orig_crops.append(model(resized).detach())

    # No optimizer needed for manual PGD
    
    print(f"DEBUG: Starting PGD. Eps={eps}, Alpha={alpha}, Steps={steps}")

    for i in range(steps):
        if x_adv.grad is not None:
             x_adv.grad.zero_()
             
        loss = 0
        valid_crops = 0
        
        # Compute adversarial embeddings
        for idx, (x_b, y_b, w_b, h_b) in enumerate(boxes):
            if idx >= len(orig_crops): break
            
            adv_crop = get_crop(x_adv, (x_b, y_b, w_b, h_b))
            if adv_crop.shape[2] == 0 or adv_crop.shape[3] == 0: continue
            
            adv_resized = F.interpolate(adv_crop, size=(160, 160), mode='bilinear', align_corners=False)
            adv_emb = model(adv_resized)
            
            # Loss: Maximize Cosine Distance
            # Cosine Similarity is [-1, 1]. 1 = Same, -1 = Opposite.
            # We want to minimize similarity.
            sim = F.cosine_similarity(adv_emb, orig_crops[idx])
            loss += sim.mean()
            valid_crops += 1
            
        if valid_crops == 0:
            break 
            
        loss.backward()
        
        # Update with Mask Constraint
        grad = x_adv.grad.data
        grad_masked = grad * mask 
        
        # Ascent: We want to MINIMIZE similarity, so we move AGAINST the gradient of similarity.
        # Loss was 'sim'. So grad is d(sim)/dx. We want to decrease sim.
        # x = x - alpha * grad
        
        x_adv.data = x_adv.data - alpha_scaled * grad_masked.sign()
        
        # Projection
        delta = torch.clamp(x_adv.data - x, -eps_scaled, eps_scaled)
        x_adv.data = torch.clamp(x + delta, -1.0, 1.0)
        
        # Debug
        if i % 2 == 0:
             print(f"DEBUG: Step {i}, Loss(Sim)={loss.item()/valid_crops:.3f}, GradMag={grad_masked.abs().mean().item():.5f}")
        
    # Final check
    dist_sum = 0
    count = 0
    with torch.no_grad():
        for idx, (x_b, y_b, w_b, h_b) in enumerate(boxes):
            if idx >= len(orig_crops): break
            adv_crop = get_crop(x_adv, (x_b, y_b, w_b, h_b))
            adv_resized = F.interpolate(adv_crop, size=(160, 160), mode='bilinear', align_corners=False)
            final_emb = model(adv_resized)
            
            # L2 dist
            d = torch.norm(final_emb - orig_crops[idx]).item()
            dist_sum += d
            count += 1
            
    avg_dist = dist_sum / count if count > 0 else 0.0
    print(f"DEBUG: Final Dist={avg_dist:.3f}")
    
    return _torch_to_pil(x_adv), avg_dist

# Alias for compatibility if needed
lite_adversarial_attack = pgd_attack_targeted
pgd_attack_embedding = pgd_attack_targeted
has_torch = has_deps
