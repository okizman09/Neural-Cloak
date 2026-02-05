import io
import time

from PIL import Image, ImageOps
import streamlit as st
import numpy as np

from cloak import cloak_image, compute_ssim, pil_save_with_exif
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
from watermark import embed_watermark, extract_watermark, generate_watermark_id

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

# Sidebar (Cleaned up)
with st.sidebar:
    st.header("Help & Info")
    st.info("""
    **How it works:**
    1. Upload a photo.
    2. Choose a protection level.
    3. The app modifies the image pixels to confuse AI.
    
    **💧 Watermark:**
    - Use the **Watermark Tools** tab to embed a hidden ID.
    - You can later **Verify** the image to prove it's yours.

    **Privacy:**
    Your photos are processed locally (if running locally) or in this session only. We do not store them.
    """)
    
    with st.expander("Advanced / Debug"):
        seed = st.number_input("Random seed", value=0)
        if seed == 0: seed = None

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


tabs = st.tabs(["🛡️ Cloak Image", "💧 Watermark Tools"])

with tabs[0]:
    st.header("Protect Your Image")
    st.markdown("Prevent AI from recognizing faces in your photos while keeping them looking natural.")
    
    uploaded = st.file_uploader("Choose an image (JPEG/PNG)", key="cloak_upload")
    img, err = validate_upload(uploaded)
    
    if err:
        st.error(err)
    
    if img is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(img, use_container_width=True)
            
        st.divider()
        st.subheader("Configuration")
        
        # Simplified Mode Selection
        mode_help = """
        **Low (Invisible):** Best for social media. Looks perfect to humans.
        **Medium (Balanced):** Good balance of protection and quality.
        **High (Maximum):** Use for sensitive photos. Might look slightly 'soft' or 'grainy'.
        """
        st.markdown(mode_help)
        
        mode = st.select_slider(
            "Select Protection Level",
            options=["Low (Invisible)", "Medium (Balanced)", "High (Maximum)"],
            value="Medium (Balanced)"
        )
        
        # Hidden params mapping
        pgd_params = {}
        if mode == "Low (Invisible)":
            pgd_params = {"eps": 4.0, "steps": 5, "quality": 0.95, "security": 0.4}
        elif mode == "Medium (Balanced)":
            pgd_params = {"eps": 8.0, "steps": 10, "quality": 0.85, "security": 0.6}
        else: # High
            pgd_params = {"eps": 16.0, "steps": 15, "quality": 0.75, "security": 0.7}
        
        if st.button("🛡️ Protect Image", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            # Step 1: Detect Faces (Backend process, hidden details unless debug)
            status_text.text("Scanning for faces...")
            progress_bar.progress(20)
            
            # Run detection (using existing backend logic)
            boxes_o_haar, _ = detect_faces_haar(img)
            boxes_o_fr, _ = detect_faces_face_recognition(img)
            boxes_o_dnn = []
            try:
                boxes_o_dnn, _ = detect_faces_dnn(img, conf_threshold=0.5)
            except:
                pass # Fail gracefully if model download fails
                
            # Combine detections
            boxes_o = boxes_o_haar + boxes_o_fr + boxes_o_dnn
            
            status_text.text(f"Found {len(boxes_o)} face(s). Applying safeguards...")
            progress_bar.progress(40)
            
            # Step 2: Apply Logic
            cloaked_img = None
            
            # Use the new Advanced Adaptive Pipeline
            st.info(f"🔒 Processing image with {mode}...")
            
            # Import pipeline (lazy load)
            try:
                from pipeline import CloakPipeline
                pipeline = CloakPipeline(device='cpu')
                
                status_log = st.empty()
                metric_display = st.empty()
                
                def update_ui(data):
                    # Simple progress update
                    status_log.text(f"Optimizing... (Step {data['attempt']})")
                        
                # Run Pipeline
                cloaked_img, final_metrics, success = pipeline.run(
                    img, 
                    security_threshold=pgd_params['security'],
                    quality_target=pgd_params['quality'],
                    max_attempts=8,
                    on_step_callback=update_ui,
                    boxes=boxes_o_haar, 
                    eps=pgd_params['eps'],
                    steps=pgd_params['steps']
                )
                
                if success:
                    status_log.success("✓ Protection Applied!")
                else:
                    status_log.warning("⚠️ Optimization limit reached (best effort used).")
                    
            except Exception as e:
                st.error(f"Failed to run protection: {e}")
                cloaked_img = cloak_image(img, strength=20.0, seed=seed)
            
            progress_bar.progress(100)
            elapsed = time.time() - start_time
            status_text.text("Done!")
            
            with col2:
                st.subheader("Protected Result")
                st.image(cloaked_img, use_container_width=True)
                
                st.divider()
                st.subheader("Status Report")
                
                # Get Security Score
                ai_dist = None
                if 'final_metrics' in locals() and final_metrics:
                    ai_dist = final_metrics.get('security', 0.0)
                elif embeddings_ok is not None:
                     ai_dist = embeddings_ok

                # Primary Status
                is_secure = False
                if ai_dist is not None and ai_dist > 1.0:
                    is_secure = True
                    st.success("✅ **SECURED: Identity Hidden**")
                    st.markdown("Your face is visible to humans, but AI sees a different person.")
                elif len(boxes_o) == 0:
                    st.success("✅ **SECURED: No Faces Found**")
                    st.markdown("No faces were detected to protect.")
                else:
                    st.warning("⚠️ **PARTIAL PROTECTION**")
                    st.markdown("The AI might still recognize you. Try a higher setting.")

                # Technical Details
                with st.expander("View Technical Details"):
                    st.markdown(f"**Processing Time:** {elapsed:.1f}s")
                    
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Visual Quality", f"{compute_ssim(img, cloaked_img):.2f}", help="1.0 is identical to original")
                    with m2:
                        val = f"{ai_dist:.2f}" if ai_dist else "N/A"
                        st.metric("AI Distance", val, help="Target > 1.0 (Different Identity)")
                    
                    st.caption("AI Distance measures how different your face looks to facial recognition models.")
                
                # Download
                exif = img.info.get("exif", None)
                data = pil_save_with_exif(cloaked_img, exif)
                st.download_button(
                    "⬇️ Download Cloaked Image", 
                    data, 
                    file_name=f"cloaked_{int(time.time())}.jpg", 
                    mime="image/jpeg",
                    type="primary"
                )

with tabs[1]:
    st.header("Watermark Tools")
    st.markdown("Apply or Verify invisible watermarks on your images.")
    
    wm_action = st.radio("Select Action:", ["Apply Watermark (Embed ID)", "Verify Watermark (Read ID)"], horizontal=True)
    
    if wm_action == "Apply Watermark (Embed ID)":
        st.subheader("Apply Invisible Watermark")
        st.markdown("Upload any image to embed a hidden, unique ID.")
        
        uploaded_apply = st.file_uploader("Upload image to watermark", key="apply_wm_up")
        img_apply, err_apply = validate_upload(uploaded_apply)
        
        if err_apply:
            st.error(err_apply)
            
        if img_apply:
            st.image(img_apply, width=300, caption="Original Image")
            
            if st.button("💧 Embed Watermark", type="primary"):
                with st.spinner("Embedding invisible signal..."):
                    # Generate ID
                    new_id = generate_watermark_id()
                    # Embed
                    watermarked_img = embed_watermark(img_apply, new_id)
                    time.sleep(0.5)
                    
                st.success("✅ Watermark applied successfully!")
                st.info(f"**Watermark ID:** `{new_id}`")
                st.caption("Save this ID. It is now hidden inside the image pixels.")
                
                # Preview
                st.image(watermarked_img, width=300, caption="Watermarked Image (Looks identical)")
                
                # Download
                exif = img_apply.info.get("exif", None)
                data = pil_save_with_exif(watermarked_img, exif)
                st.download_button(
                    "⬇️ Download Watermarked Image", 
                    data, 
                    file_name=f"watermarked_{int(time.time())}.jpg", 
                    mime="image/jpeg",
                    type="primary"
                )
    
    else:
        st.subheader("Verify Watermark")
        st.markdown("Upload an image to check if it contains a Neural Cloak invisible signature.")
        
        uploaded_wm = st.file_uploader("Upload image to verify", key="verify_wm")
        img_wm, err_wm = validate_upload(uploaded_wm)
        
        if img_wm is not None:
            st.image(img_wm, caption="Uploaded Image", width=300)
            
            if st.button("🔍 Scan for Watermark"):
                with st.spinner("Decoding hidden signature..."):
                    found_text = extract_watermark(img_wm)
                    time.sleep(1) # Dramatic pause for effect
                    
                if found_text:
                    st.balloons()
                    st.success("✅ AUTHENTICATED: Watermark Found!")
                    st.code(found_text)
                    st.caption("This signature confirms the image was processed by Neural Cloak Protocol.")
                else:
                    st.error("❌ No valid watermark found.")
                    st.warning("This image does not appear to be protected by this tool, or the signature was destroyed.")

st.markdown("---")
st.caption("Neural Cloak Protocol MVP — Privacy & Accountability Tool")
