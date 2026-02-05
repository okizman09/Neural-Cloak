from adversary import create_feature_mask, pgd_attack_targeted, load_embedding_model
from PIL import Image
import numpy as np
import torch

def debug_mask():
    img = Image.new('RGB', (500, 500), color='grey')
    
    # Manual box for test: (100, 100, 200, 200)
    msg = "Testing PGD with Manual Boxes..."
    print(msg)
    
    # Create fake model stub with same interface as FaceNet (return tensor)
    class StubModel(torch.nn.Module):
        def forward(self, x):
            # Return random embedding [Batch, 512]
            return torch.randn(x.shape[0], 512)
            
    model = StubModel()
    
    # Run attack with explicit boxes
    # This should NOT fail even if face detection would fail on a grey image
    try:
        res, score = pgd_attack_targeted(
            model, 
            img, 
            steps=1, 
            boxes=[(100, 100, 200, 200)]
        )
        print("PGD run with manual boxes: SUCCESS")
    except Exception as e:
        print(f"PGD run FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mask()
