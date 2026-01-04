import io
import time

from PIL import Image, ImageOps
import streamlit as st
import numpy as np

from cloak import cloak_image, compute_ssim, pil_save_with_exif, find_max_strength_for_ssim, find_min_strength_for_disruption, cloak_face_regions, pil_to_np, np_to_pil
from safety_scan import (
    detect_faces_haar,
    detect_faces_dnn,
    detect_faces_mtcnn,
    draw_boxes,
)
from safety_scan import (
    detect_faces_face_recognition,
    compute_face_embeddings,
    _HAS_FACEREC,
    _HAS_MTCNN,
)

# Optional adversary imports (only if torch is available)
try:
    from adversary import has_torch, has_cuda, load_embedding_model, pgd_attack_embedding
except (ImportError, AttributeError):
    # torch/facenet-pytorch not installed; stub out functions
    def has_torch():
        return False
    def has_cuda():
        return False
    def load_embedding_model(*args, **kwargs):
        raise RuntimeError("PyTorch not installed")
    def pgd_attack_embedding(*args, **kwargs):
        raise RuntimeError("PyTorch not installed")


MAX_PIXELS = 4096 * 4096
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


st.set_page_config(page_title="Cloak Protocol", layout="wide")

st.title("The Cloak Protocol — AI-Proof Image Protection")

with st.sidebar:
    st.header("Settings")
    ssim_threshold = st.slider("SSIM threshold (visual parity)", 0.80, 0.995, 0.95, step=0.005)
    strength = st.slider("Cloak strength", 0.5, 20.0, 4.0, step=0.5)
    seed = st.number_input("Random seed (optional)", value=0)
    if seed == 0:
        seed = None


def validate_upload(uploaded) -> tuple[Image.Image | None, str | None]:
    if uploaded is None:
        return None, None
    data = uploaded.getbuffer()
    if len(data) > MAX_FILE_SIZE:
        return None, f"File too large ({len(data)} bytes). Max is {MAX_FILE_SIZE} bytes."
    try:
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        return None, f"Cannot open image: {e}"
    w, h = img.size
    if w * h > MAX_PIXELS:
        return None, f"Image too large ({w}x{h}). Max pixels is {MAX_PIXELS}."
    return img, None


tabs = st.tabs(["Cloak", "Detection Test", "Safety Scan"])


with tabs[0]:
    st.header("Cloak Image")
    uploaded = st.file_uploader("Upload image (JPEG/PNG). Max 5 MB.")
    img, err = validate_upload(uploaded)
    if err:
        st.error(err)

    if img is not None:
        st.subheader("Original")
        st.image(img, width='stretch')

        auto_tune = st.checkbox("Auto-tune strength to SSIM threshold", value=False)

        if st.button("Apply Cloak"):
            start = time.time()

            if auto_tune:
                best_strength, cloaked, best_ssim = find_max_strength_for_ssim(
                    img, target_ssim=ssim_threshold, max_strength=20.0, tol=0.25, seed=seed
                )
                elapsed = time.time() - start

                st.markdown(f"**Auto-tuned strength:** {best_strength:.2f}")
                ssim = best_ssim
            else:
                cloaked = cloak_image(img, strength=strength, seed=seed)
                elapsed = time.time() - start
                ssim = compute_ssim(img, cloaked)

            st.subheader("Cloaked")
            st.image(cloaked, width='stretch')

            st.markdown(f"**SSIM:** {ssim:.4f} (threshold {ssim_threshold})")
            st.markdown(f"**Processing time:** {elapsed:.2f}s")

            if ssim < ssim_threshold:
                st.warning(
                    "SSIM below threshold — increase threshold or reduce strength to improve visual parity."
                )

            exif = img.info.get("exif", None)
            data = pil_save_with_exif(cloaked, exif)

            st.download_button(
                "Download Cloaked Image (JPEG)", data, file_name="cloaked.jpg", mime="image/jpeg"
            )


with tabs[1]:
    st.header("Detection Test (Face Recognition) — Separated from Cloak")
    uploaded2 = st.file_uploader("Upload image for detection test", key="detect_only")
    img2, err2 = validate_upload(uploaded2)
    if err2:
        st.error(err2)

    if img2 is not None:
        st.subheader("Input Image")
        st.image(img2, width='stretch')

        if st.button("Run Haar Cascade Detection", key="detect_btn"):
            boxes, elapsed = detect_faces_haar(img2)
            st.success(f"Detection finished in {elapsed:.3f}s — {len(boxes)} faces found")
            annotated = draw_boxes(img2, boxes)
            st.image(annotated, width='stretch')


with tabs[2]:
    st.header("Safety Scan: Original vs Cloaked")
    st.markdown("""
    **What Safety Scan Does:**
    1. Detects faces in your original image using multiple detectors (Haar Cascade, DNN ResNet, face_recognition, MTCNN).
    2. Applies adversarial noise to create a "cloaked" version.
    3. Re-runs face detection on the cloaked image to verify if detections are eliminated.
    4. Computes embedding distances (facial feature vectors) to ensure the cloaked faces are far from originals, disrupting deepfake AI training.
    
    **Goal:** If Safety Scan shows fewer/zero detections and high embedding distance on the cloaked image, your photo is protected from AI tools.
    """)
    
    uploaded3 = st.file_uploader("Upload image to run safety scan (single-image)", key="safety")
    img3, err3 = validate_upload(uploaded3)
    if err3:
        st.error(err3)

    if img3 is not None:
        st.subheader("Original")
        st.image(img3, width='stretch')
        # Options for targeted region cloaking
        st.write("---")
        st.markdown("**Targeted face-region cloaking**")
        target_regions = st.checkbox("Apply stronger noise to detected face regions (may help reduce detections)", value=False)
        use_dnn = st.checkbox("Include OpenCV DNN (ResNet SSD) detector (downloads ~2.6MB model)", value=True)
        dnn_conf = st.slider("DNN confidence threshold", 0.3, 0.9, 0.5, step=0.05)
        iterative = st.checkbox("Iteratively increase face-region strength until detections reduce", value=False)
        st.markdown("**Iterative tuning options**")
        max_attempts = st.number_input("Max iterative attempts", min_value=1, max_value=20, value=6)
        strength_mult = st.slider("Strength multiplier per attempt", 1.1, 3.0, 1.6, step=0.1)
        embed_threshold = st.slider("Embedding L2 threshold (requires face_recognition)", 0.2, 1.2, 0.6, step=0.05)

        st.markdown("**Maximum Protection Mode**")
        max_protection = st.checkbox("Enable Maximum Protection (binary-search for minimum noise + safety margin)", value=False)
        if max_protection:
            min_ssim_protection = st.slider("Minimum acceptable SSIM in protection mode", 0.50, 0.95, 0.70, step=0.05)
            safety_margin = st.slider("Safety margin multiplier (apply extra noise beyond minimum)", 1.0, 3.0, 2.0, step=0.1)
        else:
            min_ssim_protection = 0.70
            safety_margin = 2.0

        st.markdown("**AI Disruption Mode (Maximum Privacy)**")
        ai_disruption = st.checkbox("⚠️ Aggressive mode: Apply maximum noise to defeat AI model training (visual quality will degrade)", value=False)
        if ai_disruption:
            st.warning("⚠️ This mode applies very strong noise that may make the image look corrupted. Use this if you need maximum protection against deepfakes/AI training.")

        if st.button("Run Safety Scan"):
            # detect original with Haar and optional face_recognition
            boxes_o_haar, t_o_haar = detect_faces_haar(img3)
            boxes_o_fr, t_o_fr = detect_faces_face_recognition(img3)
            boxes_o_dnn, t_o_dnn = ([], 0.0)
            if use_dnn:
                boxes_o_dnn, t_o_dnn = detect_faces_dnn(img3, conf_threshold=dnn_conf)
            boxes_o_mtcnn, t_o_mtcnn = ([], 0.0)
            use_mtcnn = False
            if _HAS_MTCNN:
                use_mtcnn = st.checkbox("Include MTCNN detector (facenet-pytorch)", value=False)
            if use_mtcnn and _HAS_MTCNN:
                boxes_o_mtcnn, t_o_mtcnn = detect_faces_mtcnn(img3)

            # Merge boxes from detectors (simple concatenation for reporting)
            boxes_o = boxes_o_haar + boxes_o_fr + boxes_o_dnn + boxes_o_mtcnn
            t_o = t_o_haar + t_o_fr + t_o_dnn + t_o_mtcnn

            start = time.time()

            # AI Disruption Mode: Apply maximum noise regardless of SSIM or visual quality
            if ai_disruption and len(boxes_o) > 0:
                st.info("🔒 Running AI Disruption Mode (EXTREME multi-layer disruption to defeat AI vision)...")
                
                cloaked3 = img3.copy()
                attempt = 0
                max_ai_attempts = 8
                disruption_strength = 30.0  # Start much higher
                
                # Iteratively apply EXTREME disruption until detections drop to zero
                while attempt < max_ai_attempts:
                    attempt += 1
                    
                    # Apply EXTREME multi-layer disruption (defeats AI vision models)
                    cloaked3 = cloak_image(img3, strength=disruption_strength, seed=seed)
                    
                    # Check if detections reduced
                    boxes_c_haar, _ = detect_faces_haar(cloaked3)
                    boxes_c_fr, _ = detect_faces_face_recognition(cloaked3)
                    boxes_c_dnn, _ = ([], 0.0)
                    if use_dnn:
                        boxes_c_dnn, _ = detect_faces_dnn(cloaked3, conf_threshold=dnn_conf)
                    boxes_c = boxes_c_haar + boxes_c_fr + boxes_c_dnn
                    
                    st.markdown(f"Attempt {attempt}: Disruption={disruption_strength:.1f}, Detections={len(boxes_c)}/{len(boxes_o)}")
                    
                    if len(boxes_c) == 0:
                        st.success(f"✓ Face detections eliminated after {attempt} iterations!")
                        st.info("⚠️ Image now HEAVILY CORRUPTED — AI vision models should fail to describe or generate similar images.")
                        break
                    
                    # Increase disruption more aggressively
                    disruption_strength += 5.0
                
                t_cloak = time.time() - start
                boxes_c_haar, t_c_haar = detect_faces_haar(cloaked3)
                boxes_c_fr, t_c_fr = detect_faces_face_recognition(cloaked3)
                boxes_c_dnn, t_c_dnn = ([], 0.0)
                if use_dnn:
                    boxes_c_dnn, t_c_dnn = detect_faces_dnn(cloaked3, conf_threshold=dnn_conf)
                boxes_c = boxes_c_haar + boxes_c_fr + boxes_c_dnn
                t_c = t_c_haar + t_c_fr + t_c_dnn
            # Maximum Protection Mode: binary-search for minimum disruptive strength
            elif max_protection and len(boxes_o) > 0:
                def get_detections_fn(cloaked_img):
                    """Eval function for disruption search: returns (detection_count, avg_embed_dist)"""
                    c_haar, _ = detect_faces_haar(cloaked_img)
                    c_fr, _ = detect_faces_face_recognition(cloaked_img)
                    c_dnn, _ = ([], 0.0)
                    if use_dnn:
                        c_dnn, _ = detect_faces_dnn(cloaked_img, conf_threshold=dnn_conf)
                    c_mtcnn, _ = ([], 0.0)
                    if use_mtcnn and _HAS_MTCNN:
                        c_mtcnn, _ = detect_faces_mtcnn(cloaked_img)
                    boxes_c = c_haar + c_fr + c_dnn + c_mtcnn
                    det_count = len(boxes_c)

                    # compute embeddings if available
                    avg_dist = None
                    if _HAS_FACEREC and len(boxes_o) > 0:
                        enc_orig = compute_face_embeddings(img3, boxes_o)
                        enc_cloak = compute_face_embeddings(cloaked_img, boxes_o)
                        if enc_orig and enc_cloak and len(enc_orig) == len(enc_cloak):
                            dists = [float(np.linalg.norm(a - b)) for a, b in zip(enc_orig, enc_cloak)]
                            avg_dist = sum(dists) / len(dists)
                    return det_count, avg_dist

                st.info("Running Maximum Protection Mode (binary-search for minimum noise on detected faces)...")
                min_str, cloaked3, final_dets, final_embed, final_ssim = find_min_strength_for_disruption(
                    img3, get_detections_fn,
                    boxes=boxes_o,  # Pass original face boxes for region-targeted cloaking
                    min_ssim=min_ssim_protection,
                    min_embed_dist=embed_threshold,
                    max_strength=50.0,
                    max_attempts=15,
                    seed=seed
                )
                st.markdown(f"**Min strength found:** {min_str:.2f}")

                # Apply safety margin: multiply strength by safety_margin
                if safety_margin > 1.0:
                    final_strength = min_str * float(safety_margin)
                    st.markdown(f"**Final strength (with {safety_margin:.1f}x safety margin):** {final_strength:.2f}")
                    cloaked3 = cloak_image(img3, strength=final_strength, seed=seed)
                    final_ssim = compute_ssim(img3, cloaked3)
                    # re-evaluate with final image
                    final_dets, final_embed = get_detections_fn(cloaked3)

                t_cloak = time.time() - start
                boxes_c_haar, t_c_haar = detect_faces_haar(cloaked3)
                boxes_c_fr, t_c_fr = detect_faces_face_recognition(cloaked3)
                boxes_c_dnn, t_c_dnn = ([], 0.0)
                if use_dnn:
                    boxes_c_dnn, t_c_dnn = detect_faces_dnn(cloaked3, conf_threshold=dnn_conf)
                boxes_c_mtcnn, t_c_mtcnn = ([], 0.0)
                if use_mtcnn and _HAS_MTCNN:
                    boxes_c_mtcnn, t_c_mtcnn = detect_faces_mtcnn(cloaked3)
                boxes_c = boxes_c_haar + boxes_c_fr + boxes_c_dnn + boxes_c_mtcnn
                t_c = t_c_haar + t_c_fr + t_c_dnn + t_c_mtcnn
            elif target_regions and len(boxes_o) > 0:
                face_strength = strength * 2.5
                cloaked3 = cloak_face_regions(img3, boxes_o, face_strength=face_strength, seed=seed, blur_radius=31)

                # Optional white-box adversarial attack
                use_adversary = False
                if has_torch():
                    use_adversary = st.checkbox("Also run white-box adversarial attack (PyTorch + facenet-pytorch)", value=False)
                else:
                    if st.checkbox("Check white-box adversary availability (requires PyTorch)", value=False):
                        st.info("PyTorch/facenet-pytorch not detected — install to enable adversarial attack")

                if use_adversary and has_torch():
                    # Auto-select CUDA if available
                    device = 'cuda' if has_cuda() else 'cpu'
                    st.info(f"Running white-box adversary on device: {device}")
                    model = load_embedding_model(device=device)
                    adv_eps = st.slider("Adversary eps (pixel L-inf)", 1.0, 32.0, 8.0, step=1.0)
                    adv_alpha = st.slider("Adversary alpha (step)", 0.5, 8.0, 2.0, step=0.5)
                    adv_steps = st.number_input("Adversary steps", min_value=1, max_value=200, value=10)
                    cloaked_adv, adv_dist = pgd_attack_embedding(model, cloaked3, boxes_o, eps=adv_eps, alpha=adv_alpha, steps=int(adv_steps), device=device)
                    st.markdown(f"**Adversary avg embedding L2:** {adv_dist:.3f}")
                    cloaked3 = cloaked_adv

                # Optionally iterate to increase face_strength until fewer detections or embedding distance met
                attempts = 1
                prev_good = None
                prev_metrics = None
                while iterative and attempts <= int(max_attempts):
                    # evaluate ensemble detections
                    boxes_c_haar, _ = detect_faces_haar(cloaked3)
                    boxes_c_fr, _ = detect_faces_face_recognition(cloaked3)
                    boxes_c_dnn, _ = ([], 0.0)
                    if use_dnn:
                        boxes_c_dnn, _ = detect_faces_dnn(cloaked3, conf_threshold=dnn_conf)
                    boxes_c = boxes_c_haar + boxes_c_fr + boxes_c_dnn

                    # compute SSIM and embeddings (if available)
                    ssim_now = compute_ssim(img3, cloaked3)
                    avg_dist = None
                    if _HAS_FACEREC:
                        enc_orig = compute_face_embeddings(img3, boxes_o)
                        enc_cloak = compute_face_embeddings(cloaked3, boxes_o)
                        if enc_orig and enc_cloak and len(enc_orig) == len(enc_cloak):
                            dists = [float(np.linalg.norm(a - b)) for a, b in zip(enc_orig, enc_cloak)]
                            avg_dist = sum(dists) / len(dists)

                    # If this iteration meets either condition, accept
                    if len(boxes_c) < len(boxes_o) or (avg_dist is not None and avg_dist >= embed_threshold):
                        prev_good = cloaked3
                        prev_metrics = (boxes_c, ssim_now, avg_dist)
                        break

                    # If SSIM is still acceptable, remember this as a last good candidate
                    if ssim_now >= ssim_threshold:
                        prev_good = cloaked3
                        prev_metrics = (boxes_c, ssim_now, avg_dist)

                    # prepare next attempt
                    attempts += 1
                    face_strength = face_strength * float(strength_mult)
                    cloaked3 = cloak_face_regions(img3, boxes_o, face_strength=face_strength, seed=seed, blur_radius=31)

                t_cloak = time.time() - start
                # Use final detection metrics (from prev_good if we reverted)
                if prev_good is not None and prev_good is not cloaked3:
                    # Recompute boxes for prev_good
                    cloaked3 = prev_good
                    boxes_c_haar, t_c_haar = detect_faces_haar(cloaked3)
                    boxes_c_fr, t_c_fr = detect_faces_face_recognition(cloaked3)
                    boxes_c_dnn, t_c_dnn = ([], 0.0)
                    if use_dnn:
                        boxes_c_dnn, t_c_dnn = detect_faces_dnn(cloaked3, conf_threshold=dnn_conf)
                    boxes_c = boxes_c_haar + boxes_c_fr + boxes_c_dnn
                    t_c = t_c_haar + t_c_fr + t_c_dnn
                else:
                    boxes_c_haar, t_c_haar = detect_faces_haar(cloaked3)
                    boxes_c_fr, t_c_fr = detect_faces_face_recognition(cloaked3)
                    boxes_c_dnn, t_c_dnn = ([], 0.0)
                    if use_dnn:
                        boxes_c_dnn, t_c_dnn = detect_faces_dnn(cloaked3, conf_threshold=dnn_conf)
                    boxes_c = boxes_c_haar + boxes_c_fr + boxes_c_dnn
                    t_c = t_c_haar + t_c_fr + t_c_dnn
            else:
                cloaked3 = cloak_image(img3, strength=strength, seed=seed)
                t_cloak = time.time() - start
                boxes_c_haar, t_c_haar = detect_faces_haar(cloaked3)
                boxes_c_fr, t_c_fr = detect_faces_face_recognition(cloaked3)
                boxes_c_dnn, t_c_dnn = ([], 0.0)
                if use_dnn:
                    boxes_c_dnn, t_c_dnn = detect_faces_dnn(cloaked3, conf_threshold=dnn_conf)
                boxes_c = boxes_c_haar + boxes_c_fr + boxes_c_dnn
                t_c = t_c_haar + t_c_fr + t_c_dnn

            ssim = compute_ssim(img3, cloaked3)

            # embedding-based check if face_recognition available
            embeddings_ok = None
            if len(boxes_o) > 0 and _HAS_FACEREC:
                enc_orig = compute_face_embeddings(img3, boxes_o)
                enc_cloak = compute_face_embeddings(cloaked3, boxes_o)
                if enc_orig and enc_cloak and len(enc_orig) == len(enc_cloak):
                    # compute average L2 distance
                    dists = [float(np.linalg.norm(a - b)) for a, b in zip(enc_orig, enc_cloak)]
                    avg_dist = sum(dists) / len(dists)
                    embeddings_ok = avg_dist

            st.markdown(f"**Original detection (Haar+FR):** {len(boxes_o)} faces, {t_o:.3f}s")
            st.markdown(f"**Cloaked detection (Haar+FR):** {len(boxes_c)} faces, {t_c:.3f}s")
            st.markdown(f"**Cloaking time:** {t_cloak:.3f}s")
            st.markdown(f"**SSIM:** {ssim:.4f}")
            if embeddings_ok is not None:
                st.markdown(f"**Avg embedding L2 distance:** {embeddings_ok:.3f}")

            if len(boxes_c) < len(boxes_o):
                st.success("Safety Scan: Cloak reduced face detections — PASS")
            elif len(boxes_c) == 0 and len(boxes_o) > 0:
                st.success("Safety Scan: Face detection removed completely — STRONG PASS")
            else:
                st.warning("Safety Scan: Cloak did not reduce detections — consider adjusting strength or algorithm.")

            st.subheader("Cloaked Image Preview")
            st.image(cloaked3, width='stretch')

            exif = img3.info.get('exif', None)
            data = pil_save_with_exif(cloaked3, exif)
            st.download_button("Download Cloaked Image (JPEG)", data, file_name="cloaked.jpg", mime="image/jpeg")

st.markdown("---")
st.caption("Use responsibly — not for harassment or illegal activity.")
