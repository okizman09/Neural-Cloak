"""Optional white-box adversarial attacks targeting face embeddings/detectors.

This module is optional and only active if `torch` and `facenet_pytorch` are installed.
It provides a PGD-style attack that attempts to maximize embedding L2 distance
between the original and perturbed images using an embedding model (InceptionResnetV1).

Note: CPU-only attacks are slow. Use carefully.
"""
from typing import Tuple, TYPE_CHECKING

try:
    import torch
    import torch.nn.functional as F
    from facenet_pytorch import InceptionResnetV1
    _HAS_TORCH = True
except Exception:
    torch = None
    F = None
    InceptionResnetV1 = None
    _HAS_TORCH = False

from PIL import Image
import numpy as np

if TYPE_CHECKING:
    import torch as torch_type


def has_torch() -> bool:
    return _HAS_TORCH


def has_cuda() -> bool:
    if not _HAS_TORCH:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _pil_to_torch(img: Image.Image):
    # Convert PIL RGB to torch tensor normalized to [-1, 1]
    arr = np.array(img.convert('RGB')).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    tensor = tensor.to(torch.float32) / 127.5 - 1.0
    return tensor


def _torch_to_pil(tensor):
    t = tensor.detach().cpu().squeeze(0)
    arr = ((t + 1.0) * 127.5).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr)


def load_embedding_model(device: str = 'cpu'):
    if not _HAS_TORCH:
        raise RuntimeError('PyTorch or facenet_pytorch not available')
    model = InceptionResnetV1(pretrained='vggface2').eval()
    model = model.to(device)
    return model


def compute_embedding(model, pil_img: Image.Image, device: str = 'cpu'):
    t = _pil_to_torch(pil_img).to(device)
    with torch.no_grad():
        emb = model(F.interpolate(t, size=(160, 160)))
    return emb.detach()


def pgd_attack_embedding(model, orig_img: Image.Image, boxes: list[tuple], eps: float = 8.0, alpha: float = 2.0, steps: int = 10, device: str = 'cpu'):
    """Perform a simple PGD attack to maximize average embedding L2 distance for given face boxes.

    Args:
        model: embedding model (InceptionResnetV1)
        orig_img: PIL.Image RGB
        boxes: list of (x,y,w,h) face boxes — currently used to compute embeddings but attack applies to whole image
        eps: max L-inf perturbation in pixel scale (0-255)
        alpha: step size in pixel scale
        steps: PGD steps

    Returns: (perturbed_image, avg_dist)
    """
    if not _HAS_TORCH:
        raise RuntimeError('PyTorch not available')

    device = torch.device(device)
    model = model.to(device)

    orig_embs = []
    for _ in boxes:
        emb = compute_embedding(model, orig_img, device=device)
        orig_embs.append(emb)
    if not orig_embs:
        # no faces detected; return original
        return orig_img, 0.0

    x = _pil_to_torch(orig_img).to(device)
    x_adv = x.clone().detach()
    x_adv.requires_grad = True

    # convert eps/alpha to [-1,1] scale
    eps_scaled = eps / 127.5
    alpha_scaled = alpha / 127.5

    optimizer = None
    for step in range(steps):
        optimizer = torch.optim.SGD([x_adv], lr=1.0)
        optimizer.zero_grad()

        # compute embeddings on resized crop of whole image
        emb_adv = model(F.interpolate(x_adv, size=(160, 160)))
        # compute loss as negative average L2 to original (maximize distance)
        losses = []
        for emb0 in orig_embs:
            # emb0 is detached; ensure shape matches
            emb0 = emb0.to(device)
            losses.append(-F.mse_loss(emb_adv, emb0))
        loss = sum(losses) / len(losses)
        loss.backward()

        # gradient step: sign of gradient (L-inf PGD)
        grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv.detach() + alpha_scaled * grad_sign
        # project to epsilon-ball
        x_adv = torch.max(torch.min(x_adv, x + eps_scaled), x - eps_scaled)
        x_adv = torch.clamp(x_adv, -1.0, 1.0)
        x_adv.requires_grad = True

    # final embedding distance
    with torch.no_grad():
        emb_final = model(F.interpolate(x_adv, size=(160, 160)))
    dists = [float(torch.norm(emb_final - emb0.to(device)).cpu().numpy()) for emb0 in orig_embs]
    avg_dist = sum(dists) / len(dists)

    perturbed = _torch_to_pil(x_adv)
    return perturbed, avg_dist
