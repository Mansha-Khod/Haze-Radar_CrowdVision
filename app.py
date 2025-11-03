# app.py
import os
import io
import time
import requests
import base64
from datetime import datetime
from threading import Lock

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

# ---------- Config via env ----------
MODEL_GDRIVE_ID = os.environ.get("MODEL_GDRIVE_ID")  # Google Drive file id
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "crowd_vision_model.pth")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "2"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Minimal model definition (match your training architecture) ----------
class CrowdVisionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # NOTE: use the same backbone & classifier you trained with
        self.backbone = models.efficientnet_b0(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- model download (robust Google Drive helper) ----------
def download_from_gdrive(id, dest, max_retries=3, chunk_size=32768):
    """
    Download large files from Google Drive by handling confirm token.
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    for attempt in range(max_retries):
        try:
            response = session.get(URL, params={'id': id}, stream=True, timeout=30)
            token = None
            # search for confirm token in cookies or response
            for key, value in session.cookies.items():
                if key.startswith('download_warning'):
                    token = value
            if token:
                response = session.get(URL, params={'id': id, 'confirm': token}, stream=True, timeout=30)

            # If the response is HTML instead of file, still attempt to parse confirm link
            if 'Content-Type' in response.headers and 'text/html' in response.headers['Content-Type']:
                # try to find confirm token from content
                text = response.text
                import re
                m = re.search(r"confirm=([0-9A-Za-z_]+)&", text)
                if m:
                    token = m.group(1)
                    response = session.get(URL, params={'id': id, 'confirm': token}, stream=True, timeout=30)

            # Write to file
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size):
                    if chunk:
                        f.write(chunk)
            # quick check
            if os.path.getsize(dest) > 1024:
                return True
        except Exception as e:
            last_err = e
            time.sleep(2 + attempt*2)
            continue
    raise RuntimeError(f"Failed to download from Google Drive: {last_err}")

# ---------- Load model once thread-safely ----------
model = None
model_lock = Lock()

def ensure_model_loaded():
    global model
    with model_lock:
        if model is not None:
            return model

        # download if needed
        if not os.path.exists(MODEL_LOCAL_PATH):
            if not MODEL_GDRIVE_ID:
                raise RuntimeError("MODEL_GDRIVE_ID env var not set and model file not found.")
            app.logger.info("Downloading model from Google Drive...")
            download_from_gdrive(MODEL_GDRIVE_ID, MODEL_LOCAL_PATH)
            app.logger.info("Model downloaded.")

        # instantiate model and load state
        net = CrowdVisionModel(num_classes=NUM_CLASSES).to(DEVICE)

        try:
            checkpoint = torch.load(MODEL_LOCAL_PATH, map_location=DEVICE)
            # support two styles: either dict with 'model_state_dict' or raw state_dict
            state_dict = None
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # load with strict=False to avoid missing/extra keys crash
            net.load_state_dict(state_dict, strict=False)
            net.eval()
            app.logger.info("Model loaded into memory.")
        except Exception as e:
            app.logger.exception("Error loading model, re-raising.")
            raise

        model = net
        return model

# ---------- Utility: image -> tensor ----------
def pil_to_tensor(img_pil):
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    return transform(img_pil).unsqueeze(0)

# ---------- Simple visibility metrics ----------
import cv2
def calculate_visibility_score(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    contrast = float(gray.std())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / edges.size)
    brightness = float(gray.mean())
    visibility_score = min(100, (contrast * 1.5 + edge_density * 300 + (brightness - 128) * 0.2))
    visibility_score = max(0, visibility_score)
    return visibility_score, contrast, edge_density, brightness

# ---------- Health advice (example) ----------
def get_health_advice_binary(prediction, confidence, vis_score):
    # This is an example mapping — customize as you feel right
    if prediction == "clear":
        if confidence > 0.85 and vis_score > 70:
            status = "excellent"
            advice = "Air quality looks excellent. Safe to be outside."
        else:
            status = "good"
            advice = "Mostly clear. Sensitive people may monitor."
    else:
        if confidence > 0.8 and vis_score < 45:
            status = "unhealthy"
            advice = "High haze detected. Avoid outdoor activities."
        else:
            status = "moderate"
            advice = "Possible haze. Consider limiting outdoor exertion."
    return {"status": status, "advice": advice}

# ---------- Endpoints ----------
@app.route("/health", methods=["GET"])
def health():
    try:
        loaded = os.path.exists(MODEL_LOCAL_PATH)
        return jsonify({
            "status": "ok",
            "model_present": loaded,
            "device": str(DEVICE),
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ensure model loaded
        net = ensure_model_loaded()

        # accept file upload or base64 or image_url in json
        if "image" in request.files:
            image_file = request.files["image"]
            img = Image.open(image_file.stream).convert("RGB")
        else:
            data = request.get_json(force=True, silent=True) or {}
            if "image_base64" in data:
                img = Image.open(io.BytesIO(base64.b64decode(data["image_base64"]))).convert("RGB")
            elif "image_url" in data:
                resp = requests.get(data["image_url"], timeout=20)
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            else:
                return jsonify({"error": "No image provided; send multipart 'image' file or JSON with 'image_url' or 'image_base64'."}), 400

        # metrics
        img_arr = np.array(img)
        vis_score, contrast, edge_density, brightness = calculate_visibility_score(img_arr)

        # predict
        tensor = pil_to_tensor(img).to(DEVICE)
        with torch.no_grad():
            outputs = net(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])

        # map classes (adjust to your actual labels)
        label_map = {0: "clear", 1: "hazy"}
        prediction = label_map.get(pred_idx, str(pred_idx))
        health = get_health_advice_binary(prediction, conf, vis_score)

        result = {
            "success": True,
            "prediction": prediction,
            "confidence": round(conf * 100, 2),
            "visibility_score": round(vis_score, 2),
            "metrics": {"contrast": round(contrast, 2), "edge_density": round(edge_density, 3), "brightness": round(brightness, 2)},
            "health": health,
            "timestamp": datetime.utcnow().isoformat()
        }
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Predict error")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        net = ensure_model_loaded()
        data = request.get_json(force=True)
        imgs = data.get("images", [])
        if not isinstance(imgs, list) or len(imgs) == 0:
            return jsonify({"error": "Provide JSON: {images: [ {image_url:...} | {image_base64:...} ... ] }"}), 400
        results = []
        for i, item in enumerate(imgs):
            try:
                if "image_base64" in item:
                    img = Image.open(io.BytesIO(base64.b64decode(item["image_base64"]))).convert("RGB")
                elif "image_url" in item:
                    resp = requests.get(item["image_url"], timeout=20)
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                else:
                    raise ValueError("No image data in item")
                img_arr = np.array(img)
                vis_score, _, _, _ = calculate_visibility_score(img_arr)
                tensor = pil_to_tensor(img).to(DEVICE)
                with torch.no_grad():
                    outputs = net(tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    pred_idx = int(np.argmax(probs))
                    conf = float(probs[pred_idx])
                label_map = {0: "clear", 1: "hazy"}
                results.append({
                    "id": item.get("id", i),
                    "prediction": label_map.get(pred_idx, str(pred_idx)),
                    "confidence": round(conf * 100, 2),
                    "visibility_score": round(vis_score, 2)
                })
            except Exception as e:
                results.append({"id": item.get("id", i), "error": str(e)})
        return jsonify({"success": True, "results": results, "timestamp": datetime.utcnow().isoformat()})
    except Exception as e:
        app.logger.exception("Batch predict error")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Use Flask dev server for local testing; in production Render will call Gunicorn.
    app.run(host="0.0.0.0", port=port)





















