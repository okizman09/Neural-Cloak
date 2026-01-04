#!/usr/bin/env python3
"""
Simple test script to verify that AI Disruption Mode actually disrupts face embeddings.
This uses face_recognition to compare original vs cloaked embeddings.
"""

import sys
import numpy as np
from PIL import Image
from safety_scan import detect_faces_haar, compute_face_embeddings, _HAS_FACEREC, _HAS_FACENET

def test_disruption(original_path, cloaked_path):
    """Compare face embeddings between original and cloaked images."""
    
    print("=" * 70)
    print("FACE DISRUPTION TEST")
    print("=" * 70)
    
    # Load images
    try:
        orig_img = Image.open(original_path).convert("RGB")
        cloak_img = Image.open(cloaked_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return
    
    print(f"\n✓ Loaded original: {original_path}")
    print(f"✓ Loaded cloaked:  {cloaked_path}")
    
    # Detect faces
    print("\n[Step 1] Detecting faces...")
    boxes_orig, _ = detect_faces_haar(orig_img)
    boxes_cloak, _ = detect_faces_haar(cloak_img)
    
    print(f"  Original image: {len(boxes_orig)} faces detected")
    print(f"  Cloaked image:  {len(boxes_cloak)} faces detected")
    
    if len(boxes_cloak) == 0 and len(boxes_orig) > 0:
        print("  ✓ PASS: Face detection eliminated in cloaked image!")
    elif len(boxes_cloak) < len(boxes_orig):
        print("  ⚠️  PARTIAL: Some faces still detected in cloaked image")
    else:
        print("  ❌ FAIL: Same or more faces detected in cloaked image")
    
    # Compute embeddings if faces found
    print("\n[Step 2] Computing face embeddings...")
    if len(boxes_orig) == 0:
        print("  ⚠️  No faces detected in original image — skipping embeddings.")
    elif len(boxes_cloak) == 0:
        print("  ⚠️  No faces detected in cloaked image — cannot compute embeddings.")
    else:
        # compute embeddings using safety_scan's backend(s)
        enc_orig = compute_face_embeddings(orig_img, boxes_orig)
        enc_cloak = compute_face_embeddings(cloak_img, boxes_cloak)

        backend = None
        if _HAS_FACEREC:
            backend = 'face_recognition (dlib)'
        elif _HAS_FACENET:
            backend = 'facenet-pytorch (InceptionResnetV1)'
        else:
            backend = 'none'

        print(f"  Embedding backend: {backend}")

        if not enc_orig or not enc_cloak:
            print("  ⚠️  Embeddings not available (missing backend or detection mismatch).")
        else:
            # Compare pairwise up to min(len(enc_orig), len(enc_cloak))
            n = min(len(enc_orig), len(enc_cloak))
            dists = [float(np.linalg.norm(enc_orig[i] - enc_cloak[i])) for i in range(n)]
            avg_dist = sum(dists) / len(dists)
            print(f"  Embedding L2 distances (first {n} faces): {[round(d,3) for d in dists]}")
            print(f"  Avg embedding L2 distance: {avg_dist:.3f}")
            print("    (Same face: ~0.0-0.1, Different face: >0.5, Unrecognizable: >1.0)")

            if avg_dist > 1.0:
                print("  ✓ PASS: Embeddings drastically different (AI can't recognize face)!")
            elif avg_dist > 0.5:
                print("  ⚠️  PARTIAL: Embeddings different but still somewhat similar")
            else:
                print("  ❌ FAIL: Embeddings still very similar (AI could still use this)")
    
    # Visual quality check
    print("\n[Step 3] Visual quality comparison...")
    from cloak import compute_ssim
    ssim = compute_ssim(orig_img, cloak_img)
    print(f"  SSIM score: {ssim:.4f}")
    print(f"    (1.0 = identical, 0.0 = completely different)")
    
    if ssim < 0.5:
        print("  ✓ Image heavily corrupted (good for AI resistance)")
    elif ssim < 0.7:
        print("  ⚠️  Image moderately corrupted")
    else:
        print("  ❌ Image too similar to original (insufficient disruption)")
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("""
✓ If detection is eliminated AND embedding distance > 1.0:
  → AI vision models should fail to describe or generate similar images
  → Good protection against deepfakes

❌ If detection still occurs AND embedding distance < 0.5:
  → Need more aggressive cloaking
  → Try increasing disruption strength in AI Disruption Mode
    """)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_disruption.py <original_image> <cloaked_image>")
        print("\nExample:")
        print("  python test_disruption.py original.jpg cloaked.jpg")
        sys.exit(1)
    
    test_disruption(sys.argv[1], sys.argv[2])
