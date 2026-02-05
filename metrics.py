import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image

# Import local FaceNet
try:
    import torch
    from facenet_pytorch import InceptionResnetV1
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

class CloakMetrics:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        if _HAS_TORCH:
            try:
                self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            except Exception as e:
                print(f"Metrics init failed: {e}")
        
    def compute_security_score(self, original: Image.Image, cloaked: Image.Image) -> float:
        """
        Computes Security Score based on Embedding Distance.
        Score = L2 Distance (roughly 0.0 to 1.5).
        Target > 1.0 means different identity.
        """
        if self.model is None:
            return 0.0 # Fail safe
            
        # 1. Detect faces (Standard Haar for extraction)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        orig_gray = np.array(original.convert('L'))
        boxes = face_cascade.detectMultiScale(orig_gray, 1.1, 4)
        
        if len(boxes) == 0:
            return 1.0 # Already safe
            
        total_dist = 0
        count = 0
        
        # Preprocess
        def get_tensor(img_pil):
             # Resize to 160x160? No, need crop first
             t = torch.from_numpy(np.array(img_pil)).permute(2,0,1).float()
             t = (t - 127.5) / 128.0
             return t.unsqueeze(0).to(self.device)

        # We need to extract the SAME boxes from both images to compare embeddings
        # (Assuming alignment is preserved)
        
        with torch.no_grad():
            for (x, y, w, h) in boxes:
                # Crop Original
                # Resize to 160x160
                crop_orig = original.crop((x, y, x+w, y+h)).resize((160, 160))
                t_orig = get_tensor(crop_orig)
                emb_orig = self.model(t_orig)
                
                # Crop Cloaked (Same coordinates)
                crop_cloak = cloaked.crop((x, y, x+w, y+h)).resize((160, 160))
                t_cloak = get_tensor(crop_cloak)
                emb_cloak = self.model(t_cloak)
                
                # Distance
                d = torch.dist(emb_orig, emb_cloak).item()
                total_dist += d
                count += 1
                
        if count == 0: return 0.0
        return total_dist / count

    def compute_quality_score(self, original: Image.Image, cloaked: Image.Image) -> dict:
        """
        Computes Quality Score using SSIM.
        """
        # SSIM
        ssim_val = compare_ssim(
            np.array(original.convert('L')), 
            np.array(cloaked.convert('L')), 
            data_range=255
        )
        
        return {
            "ssim": ssim_val,
            "lpips": 0.0, 
            "score": ssim_val 
        }
