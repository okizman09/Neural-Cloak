# Neural Cloak Protocol 🛡️

**AI-Proof Image Privacy Tool**

Neural Cloak injects invisible, adversarial noise into your photos to prevent AI facial recognition systems (like Clearview AI, FaceNet, etc.) from identifying you, while keeping your photos looking natural to humans.

---

## Features

### 🔒 Advanced Identity Cloaking
Uses **Projected Gradient Descent (PGD)** to attack FaceNet embeddings directly.
- **Low (Invisible)**: Best for social media. Zero visual distortion.
- **Medium (Balanced)**: Good protection, high quality.
- **High (Maximum)**: Strongest protection. AI sees a completely different person.

### 💧 Invisible Watermarking
Embeds a hidden, unique ID into your image pixels.
- Use this to prove ownership or track deepfakes.
- Watermarks survive compression and resizing.

### 🛡️ Privacy First
- **Client-Side Processing**: All AI calculations happen on your machine.
- **No Data Storage**: Images are processed in memory and never saved to a cloud server.

---

## Installation

### Prerequisites
- Python 3.10+ (Tested on 3.13)
- 4GB RAM recommended

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/neural-cloak.git
   cd neural-cloak
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install PyTorch (~2GB download).*

3. **Run the App:**
   ```bash
   streamlit run app.py
   ```

---

## Usage Guide

### How to Protect an Image
1. Open the app in your browser.
2. Upload a JPEG or PNG image.
3. Select your **Protection Level** (Start with *Low*).
4. Click **Protect Image**.
5. Wait for the "Optimization Loop" to finish.
6. Look for **"✅ SECURED: Identity Hidden"** in the status report.
7. Download your protected image.

### Understanding the Report
- **Visual Quality (SSIM)**: How much the image looks like the original. 1.0 = Perfet.
- **AI Distance**: How different the AI thinks you are. Target is **> 1.0**.

### Watermarking
1. Go to the **Watermark Tools** tab.
2. Upload an image and click **Embed Watermark**.
3. Save the generated **Watermark ID**.
4. To verify later, upload the image to the **Verify** tab.

---

## Troubleshooting
- **First Run Slowness**: The app downloads the FaceNet AI model on the first run. This takes 1-2 minutes.
- **Result is "Weak"**: Try increasing the protection level to "High".
- **Crash on Start**: Ensure you have `torch` installed correctly: `pip install torch torchvision`.

---

## Credits
- **Engine**: PyTorch & FaceNet
- **Method**: Adversarial Examples (PGD)
- **UI**: Streamlit
