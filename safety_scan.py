import time
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import requests
import time

# Optional face_recognition support (embeddings + alternate detector)
try:
    import face_recognition
    _HAS_FACEREC = True
except Exception:
    face_recognition = None
    _HAS_FACEREC = False

# Optional facenet-pytorch MTCNN detector
try:
    from facenet_pytorch import MTCNN
    _HAS_MTCNN = True
except Exception:
    MTCNN = None
    _HAS_MTCNN = False

# Optional facenet-pytorch embedding model (InceptionResnetV1) as fallback for embeddings
try:
    from facenet_pytorch import InceptionResnetV1
    import torch
    _HAS_FACENET = True
except Exception:
    InceptionResnetV1 = None
    torch = None
    _HAS_FACENET = False


# OpenCV DNN (ResNet SSD) model files and download helpers
_MODEL_DIR = Path(__file__).parent / "models"
_PROTOTXT = _MODEL_DIR / "deploy.prototxt"
_CAFFEMODEL = _MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# Official OpenCV model urls (kept small and public)
_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def _ensure_dnn_model():
    """Ensure DNN model files exist locally; download if missing."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not _PROTOTXT.exists():
        # download prototxt
        r = requests.get(_PROTO_URL, timeout=30)
        r.raise_for_status()
        _PROTOTXT.write_bytes(r.content)
    if not _CAFFEMODEL.exists():
        # download caffemodel
        r = requests.get(_MODEL_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(_CAFFEMODEL, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def detect_faces_dnn(pil_img: Image.Image, conf_threshold: float = 0.5) -> Tuple[List[Tuple[int, int, int, int]], float]:
    """Detect faces using OpenCV DNN (ResNet SSD). Returns boxes and elapsed time.

    Boxes are (x, y, w, h).
    Automatically downloads model files to `models/` if missing.
    """
    try:
        _ensure_dnn_model()
    except Exception:
        # If download fails for any reason, return empty
        return [], 0.0

    net = cv2.dnn.readNetFromCaffe(str(_PROTOTXT), str(_CAFFEMODEL))

    img = pil_to_cv2(pil_img)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    start = time.time()
    detections = net.forward()
    elapsed = time.time() - start

    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            x = max(0, startX)
            y = max(0, startY)
            bw = min(w - x, endX - startX)
            bh = min(h - y, endY - startY)
            boxes.append((x, y, bw, bh))

    return boxes, elapsed


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    # PIL uses RGB, OpenCV uses BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def detect_faces_haar(pil_img: Image.Image, scaleFactor: float = 1.1, minNeighbors: int = 5) -> Tuple[List[Tuple[int, int, int, int]], float]:
    """Detect faces using OpenCV Haar Cascade. Returns boxes and elapsed time.

    Boxes are (x, y, w, h).
    """
    img = pil_to_cv2(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    start = time.time()
    rects = cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    elapsed = time.time() - start

    boxes = [tuple(map(int, r)) for r in rects]
    return boxes, elapsed


def draw_boxes(pil_img: Image.Image, boxes: List[Tuple[int, int, int, int]], color=(0, 255, 0)) -> Image.Image:
    img = pil_to_cv2(pil_img)
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    # convert back to PIL RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def detect_faces_face_recognition(pil_img: Image.Image) -> Tuple[List[Tuple[int, int, int, int]], float]:
    """Detect faces using face_recognition (HOG/CNN). Returns boxes (x,y,w,h) and elapsed time.

    If `face_recognition` isn't installed, returns empty list and 0 time.
    """
    if not _HAS_FACEREC:
        return [], 0.0

    start = time.time()
    arr = np.array(pil_img.convert("RGB"))
    # face_recognition returns (top, right, bottom, left)
    locations = face_recognition.face_locations(arr)
    elapsed = time.time() - start

    boxes = []
    for (top, right, bottom, left) in locations:
        x = left
        y = top
        w = right - left
        h = bottom - top
        boxes.append((x, y, w, h))
    return boxes, elapsed


def detect_faces_mtcnn(pil_img: Image.Image) -> Tuple[List[Tuple[int, int, int, int]], float]:
    """Detect faces using facenet_pytorch MTCNN. Returns boxes and elapsed time.

    If MTCNN isn't available, returns empty list and 0 time.
    """
    if not _HAS_MTCNN:
        return [], 0.0

    mtcnn = MTCNN(keep_all=True)
    start = time.time()
    boxes, _ = mtcnn.detect(pil_img)
    elapsed = time.time() - start
    if boxes is None:
        return [], elapsed
    out = []
    for (x1, y1, x2, y2) in boxes:
        x = int(max(0, x1))
        y = int(max(0, y1))
        bw = int(max(0, x2 - x1))
        bh = int(max(0, y2 - y1))
        out.append((x, y, bw, bh))
    return out, elapsed


def compute_face_embeddings(pil_img: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """Compute face embeddings using face_recognition for given boxes.

    Returns list of 128-d numpy arrays. If package not available, returns [].
    """
    # Prefer face_recognition if available (128-d embeddings)
    if _HAS_FACEREC:
        arr = np.array(pil_img.convert("RGB"))
        # Convert boxes into face_recognition (top,right,bottom,left)
        locations = [(y, x + w, y + h, x) for (x, y, w, h) in boxes]
        encodings = face_recognition.face_encodings(arr, known_face_locations=locations)
        return encodings

    # Fallback: use facenet-pytorch InceptionResnetV1 (512-d embeddings)
    if _HAS_FACENET and InceptionResnetV1 is not None:
        try:
            model = InceptionResnetV1(pretrained='vggface2').eval()
            # move to cpu (we assume CPU-only for this workspace)
            if torch is not None:
                device = torch.device('cpu')
                model = model.to(device)

            encs = []
            arr = np.array(pil_img.convert('RGB'))
            for (x, y, w, h) in boxes:
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(arr.shape[1], x + w)
                y2 = min(arr.shape[0], y + h)
                crop = arr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # Resize crop to 160x160 expected by InceptionResnetV1
                img_pil = Image.fromarray(crop).resize((160, 160))
                # normalize to [-1,1]
                img_arr = np.array(img_pil).astype(np.float32)
                tensor = (torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 127.5) - 1.0
                with torch.no_grad():
                    emb = model(tensor)
                encs.append(emb.cpu().numpy().reshape(-1))
            return encs
        except Exception:
            return []

    # no embedding support available
    return []
