import os
import io
import base64
import torch
import numpy as np
from PIL import Image
import cv2
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)
CORS(app)

device = torch.device("cpu")

# ==========================================================
# MODEL DEFINITION (must match training)
# ==========================================================
class CrowdVisionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CrowdVisionModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ==========================================================
# LOAD MODEL
# ==========================================================
model_path = "crowdvision_final_ohaze.pth"

model = CrowdVisionModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()

print("✅ Model loaded successfully")


# ==========================================================
# TRANSFORM (must match training)
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ==========================================================
# VISIBILITY CALC
# ==========================================================
def calculate_visibility_score(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    contrast = gray.std()
    brightness = np.mean(gray)
    edges = cv2.Canny(gray, 80, 180)
    edge_density = np.sum(edges > 0) / edges.size

    score = (contrast * 0.4) + (edge_density * 3200 * 0.4) + (brightness * 0.2)
    score = np.clip(score, 0, 100)

    return score, contrast, edge_density, brightness


# ==========================================================
# PREDICTION ROUTE
# ==========================================================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Accept Base64 OR File Upload
        if "image" in request.files:
            image = Image.open(request.files["image"].stream).convert("RGB")
        else:
            data = request.get_json()
            img_data = base64.b64decode(data["image_base64"])
            image = Image.open(io.BytesIO(img_data)).convert("RGB")

        # Visibility Metrics
        visibility_score, contrast, edge_density, brightness = calculate_visibility_score(image)

        # Model Prediction
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_tensor)
            probs = torch.softmax(out, dim=1)
            confidence, pred = torch.max(probs, 1)

        confidence_value = round(confidence.item() * 100, 2)
        label_map = {0: "clear", 1: "hazy"}
        prediction = label_map[pred.item()]

        # ======================================================
        # SKY + CLOUD CORRECTION (fix misclassification on blue sky)
        # ======================================================
        img_np = np.array(image)
        avg_r = np.mean(img_np[:, :, 0])
        avg_g = np.mean(img_np[:, :, 1])
        avg_b = np.mean(img_np[:, :, 2])

        blue_dominant = (avg_b > avg_g + 15) and (avg_b > avg_r + 15)
        white_clouds_present = np.mean(img_np) > 180

        if blue_dominant and white_clouds_present:
            prediction = "clear"
            visibility_score = min(100, visibility_score + 25)
            confidence_value = max(confidence_value, 85)

        # ======================================================
        # VISIBILITY CALIBRATION
        # ======================================================
        if prediction == "clear":
            visibility_score = min(100, visibility_score * 1.3)
        else:
            visibility_score = max(0, visibility_score * 0.8)

        visibility_score = round(float(visibility_score), 2)

        # ======================================================
        # RESPONSE
        # ======================================================
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": confidence_value,
            "visibility_score": visibility_score,
            "metrics": {
                "contrast": round(float(contrast), 2),
                "edge_density": round(float(edge_density * 100), 2),
                "brightness": round(float(brightness), 2)
            },
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/")
def home():
    return jsonify({
        "message": "🌤️ HazeRadar API is running!",
        "predict": "/predict (POST)"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)











































