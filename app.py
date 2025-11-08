import os
import io
import base64
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import models, transforms

from flask import Flask, request, jsonify
from flask_cors import CORS
import gdown


# ===============================================================
# Flask Setup
# ===============================================================
app = Flask(__name__)
CORS(app)

device = torch.device("cpu")


# ===============================================================
# Download Model from Google Drive if NOT already present
# ===============================================================
MODEL_PATH = "crowd_vision_model.pth"
DRIVE_FILE_ID = "1PLoMmldxg7QZKONrb29CVKtqB9_kZjac"

if not os.path.exists(MODEL_PATH):
    print(" Downloading model from Google Drive (direct mode)...")
    gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
else:
    print(" Model already present.")



# ===============================================================
# Model Architecture (Matches Training Exactly)
# ===============================================================
class CrowdVisionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CrowdVisionModel, self).__init__()
       
        self.backbone = models.efficientnet_b0(weights=None)

       
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


model = CrowdVisionModel(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()

print(" Model loaded successfully!")

# ===============================================================
# Image Preprocessing
# ===============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])


# ===============================================================
# Visibility Calculation
# ===============================================================
def calculate_visibility_score(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    contrast = gray.std()
    brightness = np.mean(gray)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    visibility = (contrast * 0.4) + (edge_density * 4000 * 0.4) + (brightness * 0.2)
    visibility = np.clip(visibility, 0, 100)

    return visibility, contrast, edge_density, brightness


# ===============================================================
# Predict Endpoint
# ===============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' in request.files:  
            image_file = request.files['image']
            image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        else:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        vis_score, contrast, edges, brightness = calculate_visibility_score(image)
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        # Calculate probabilities
        clear_prob = probs[0][0].item()
        hazy_prob = probs[0][1].item()
        
        # Smart uncertainty detection
        if abs(clear_prob - hazy_prob) < 0.2:  
            prediction = "uncertain"
            confidence_value = max(clear_prob, hazy_prob) * 100
        else:
            label_map = {0: "clear", 1: "hazy"}
            prediction = label_map[predicted.item()]
            confidence_value = float(confidence.item() * 100)
        
        # Convert to percentages for debugging
        clear_percent = clear_prob * 100
        hazy_percent = hazy_prob * 100
        
        print(f" DEBUG - Clear: {clear_percent:.1f}% | Hazy: {hazy_percent:.1f}% | Predicted: {predicted.item()}")
        print(f" DEBUG - Final Prediction: {prediction} | Confidence: {confidence_value:.1f}%")

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence_value, 2),
            "visibility_score": round(vis_score, 2),
            "probabilities": {
                "clear": round(clear_percent, 2),
                "hazy": round(hazy_percent, 2)
            },
            "metrics": {
                "contrast": round(contrast, 2),
                "edge_density": round(edges * 100, 2),
                "brightness": round(brightness, 2)
            }
        })

    except Exception as e:
        print(f"❌ Error in /predict: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/')
def home():
    return {"status": "CrowdVision HazeRadar API Running", "model": "O-HAZE Trained"}


# ===============================================================
# Run App
# ===============================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)














