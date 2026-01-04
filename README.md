# Cloak Protocol — AI-Proof Image Protection

This repository contains a Streamlit MVP for The Cloak Protocol — a privacy-first, local tool that applies invisible adversarial perturbations to images to disrupt AI face-detection and deepfake pipelines.

## Features (MVP)
- **Local, in-memory image processing** (no uploads to servers)
- **Streamlit UI** for single-image processing
- **Cloak engine** (`cloak.py`) with Gaussian-based perturbation + aggressive disruption
- **Safety scan** (`safety_scan.py`) using multiple detectors (Haar Cascade, DNN, face_recognition, MTCNN)
- **AI Disruption Mode** — extreme multi-layer noise to defeat ChatGPT, DALL-E, and vision models
- **EXIF metadata** preservation when downloading cloaked image

## Quickstart

1. Create and activate a Python environment (Windows PowerShell example):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

2. Open browser to `http://localhost:8501`

3. Go to **Safety Scan** tab

4. Enable **"⚠️ Aggressive mode: Apply maximum noise to defeat AI model training"**

5. Upload image and click **"Run Safety Scan"**

6. Download the cloaked image

## Testing Your Cloaked Image

### Method 1: Local Verification (No External Sites)
```powershell
python test_disruption.py original.jpg cloaked.jpg
```
This compares face embeddings to verify disruption worked.

### Method 2: External Testing

**Face Detection Testing:**
- [Nyckel Face Detector](https://www.nyckel.com/pretrained-classifiers/face-detector) — Should detect 0 faces
- [TensorFlow COCO-SSD](https://coco-ssd.glitch.me/) — Should not detect faces

**AI Vision Testing:**
- [ChatGPT](https://chat.openai.com/) — Upload and ask: "Can you describe the person in this image?" 
  - ✅ Expected: "I cannot identify people in images" or fails to analyze
  - ❌ Bad: Describes person details (clothing, expression, etc.)

- [Google Lens](https://lens.google.com/) — Reverse-image search should fail or return irrelevant results

**Success Criteria:**
- ✅ Face detection returns 0 detections on cloaked image
- ✅ ChatGPT/DALL-E cannot describe or generate similar images
- ✅ Reverse-image search fails (Google Lens, TinEye)
- ✅ Embedding distance > 1.0 (original vs cloaked face is unrecognizable)

## Usage Notes
- Max resolution: 4096×4096 pixels
- Max file size: 5 MB
- CPU-only processing for MVP (GPU optional: install PyTorch)
- Language: English
- License: MIT

## Modes Explained

### 1. Cloak Tab
- Simple Gaussian noise with automatic SSIM tuning
- Preserves visual quality (SSIM > 0.95 by default)
- **Use when:** You want a lightly cloaked image

### 2. Detection Test Tab
- Runs face detection only (no cloaking)
- Shows detected faces on original image
- **Use when:** You want to see what detectors find

### 3. Safety Scan Tab with Modes

#### Standard Mode (Default)
- Applies balanced noise with SSIM preservation
- **Use when:** You want mild protection

#### Maximum Protection Mode
- Binary-search to find minimum noise needed
- Applies extra safety margin (2.0x default)
- Relaxed SSIM constraint (0.70 min)
- **Use when:** You want moderate protection

#### ⚠️ AI Disruption Mode (Most Aggressive)
- Applies EXTREME multi-layer noise:
  - Extreme high-frequency noise (defeats feature extraction)
  - Aggressive color shifts (defeats color analysis)
  - Heavy block corruption (defeats pattern recognition)
  - Pixel scrambling (defeats spatial correlation)
  - Frequency-domain disruption (defeats DCT/FFT analysis)
  - Cross-channel mixing (defeats RGB analysis)
- Iteratively increases noise until face detection eliminated
- **Use when:** ChatGPT/DALL-E are still working on your image (maximum privacy)
- ⚠️ **Warning:** Image will be heavily corrupted and may look like noise

## AI Disruption Algorithm Details

The aggressive disruption applies **6 simultaneous attack layers**:

1. **Extreme High-Frequency Noise** (σ = strength × 1.2)
   - Defeats CNN feature extraction
   - Breaks learned texture patterns

2. **Aggressive Color Channel Randomization** (±100 per channel)
   - Defeats color-based analysis
   - Breaks RGB correlation

3. **Heavy Block Corruption** (20-80 blocks, strong noise)
   - Defeats pattern recognition
   - Unpredictable spatial patterns

4. **Pixel-Level Scrambling** (random shuffling in patches)
   - Defeats spatial correlation detection
   - Breaks neighborhood relationships

5. **Frequency-Domain Disruption** (phase shifts in DCT/spatial tiles)
   - Defeats Fourier-based analysis
   - Breaks frequency patterns

6. **Cross-Channel Mixing** (random RGB channel swaps)
   - Defeats independent channel analysis
   - Breaks color space assumptions

These layers work together to defeat modern vision transformers and diffusion models.

## Disclaimer

Use responsibly — not for harassment, illegal activity, or privacy invasion. This tool is meant to protect consenting individuals from non-consensual exploitation (deepfakes, AI training without consent).

## Optional Features

Install optional dependencies for more detectors and GPU support:

```powershell
pip install -r requirements-optional.txt
```

This adds:
- PyTorch + facenet-pytorch (white-box adversarial attacks, GPU support)
- face_recognition (HOG/CNN face detection)
- MTCNN detector (more accurate face detection)

⚠️ **Note:** Windows Python 3.13 has build issues. Use Python 3.11 or Docker if installation fails.

## Credits

Built for Power Hacks 2025 — Safety by Design / Women in Public Life.
