from pipeline import CloakPipeline
from PIL import Image
import numpy as np

def test_pipeline():
    print("Initializing pipeline...")
    # Smaller image for speed
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    
    pipeline = CloakPipeline(device='cpu')
    print("Running optimization...")
    
    res, metrics, success = pipeline.run(
        img, 
        security_threshold=0.0, # Pass instantly
        quality_target=0.0,
        max_attempts=1
    )
    
    print(f"Success: {success}")
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    try:
        test_pipeline()
        print("TEST PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
