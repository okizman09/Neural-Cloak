from typing import Tuple
from PIL import Image
import torch
import adversary
from metrics import CloakMetrics

class CloakPipeline:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None 
        self.metrics = None
    
    def _init_models(self):
        if self.model is None:
            if not adversary.has_deps():
                raise RuntimeError("Dependency Error: Torch not found.")
            
            self.model = adversary.load_embedding_model(self.device)
            self.metrics = CloakMetrics(self.device)

    def run(self, 
            image: Image.Image, 
            security_threshold: float = 0.6, 
            quality_target: float = 0.8,
            max_attempts: int = 8,
            on_step_callback=None,
            boxes: list = None,
            # PGD Params
            eps: float = 16.0,
            alpha: float = 4.0,
            steps: int = 5
            ) -> Tuple[Image.Image, dict, bool]:
        """
        Runs Adaptive PGD Cloaking.
        """
        self._init_models()
        
        # Initial parameters (now passed in or default)
        
        best_image = image
        best_metrics = {
            "security": 0.0,
            "quality": 1.0,
            "ssim": 1.0,
            "lpips": 0.0
        }
        success = False
        
        for attempt in range(1, max_attempts + 1):
            # Dynamic Alpha: Ensure we can reach the boundary of eps in the given steps
            # scale factor 1.5 to overshoot slightly for faster convergence
            alpha = (eps / steps) * 1.5
            
            # Run PGD
            perturbed, avg_dist = adversary.pgd_attack_targeted(
                self.model, image, eps=eps, alpha=alpha, steps=steps, device=self.device, boxes=boxes
            )
            
            # Verify
            sec_score = avg_dist # From PGD directly is faster, but verify properly
            # Re-verify with metrics (independent check)
            # Actually pgd_attack returns the avg_dist calculated during attack, which is accurate for that model.
            
            # Quality
            qual_metrics = self.metrics.compute_quality_score(image, perturbed)
            qual_score = qual_metrics['score']
            
            current_metrics = {
                "security": avg_dist,
                "quality": qual_score,
                "ssim": qual_metrics['ssim'],
                "lpips": 0.0
            }
            
            # Check thresholds
            sec_pass = avg_dist >= security_threshold
            qual_pass = qual_score >= quality_target
            
            # UI Update
            if on_step_callback:
                on_step_callback({
                    "attempt": attempt,
                    "eps": eps,
                    "metrics": current_metrics,
                    "pass": (sec_pass and qual_pass)
                })
            
            # Success?
            if sec_pass and qual_pass:
                best_image = perturbed
                best_metrics = current_metrics
                success = True
                break
                
            # Logic to adjust
            if not sec_pass:
                # Need more perturbation
                eps += 4.0
                alpha += 0.5
                steps += 5
            elif not qual_pass:
                # Too much perturbation?
                eps *= 0.8
                # Retry with lower eps next time
                
            # Keep best
            if avg_dist > best_metrics['security']:
                best_image = perturbed
                best_metrics = current_metrics
                
            # Cap
            if eps > 32.0:
                eps = 32.0 # Max visual distortion limit
            
        return best_image, best_metrics, success
