import os
import io
import torch
import torch.nn as nn
import numpy as np
import cv2
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import models, transforms
from PIL import Image

# =========================================================
# FLASK SETUP
# =========================================================
app = Flask(__name__)
CORS(app)

device = torch.device("cpu")
torch.set_num_threads(1)

# =========================================================
# MODEL ARCHITECTURE (must match training)
# =========================================================
class CrowdVisionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CrowdVisionModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)



# =========================================================
# LOAD TRAINED WEIGHTS
# =========================================================
import gdown

MODEL_PATH = "crowd_vision_model.pth"
FILE_ID = "1PLoMmldxg7QZKONrb29CVKtqB9_kZjac"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully!")
else:
    print("✓ Model already exists, skipping download.")

device = torch.device("cpu")

model = CrowdVisionModel(num_classes=2).to(device)
model.load_state_dict(torch.load("crowd_vision_model.pth", map_location=device))
model.eval()

print("✅ Model loaded successfully")


# =========================================================
# IMAGE TRANSFORM (MUST MATCH TRAINING TRANSFORMS)
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================================================
# VISIBILITY SCORE HELPERS
# =========================================================
def calculate_visibility_score(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Dark Channel Prior (strong haze indicator)
    dark_channel = cv2.erode(np.min(image_cv, axis=2),
                             cv2.getStructuringElement(cv2.MORPH_RECT, (15,15)))
    haze_strength = np.mean(dark_channel)  # higher = more haze

    # Contrast (lower in haze)
    contrast = gray.std()

    # Edge density (lower in haze)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # Convert into a final readable 0–100 score
    score = (
        (contrast * 0.7) +         # clarity
        (edge_density * 120) -     # sharpness factor
        (haze_strength * 0.05)     # haze reduction factor
    )

    visibility_score = np.clip(score, 0, 100)

    return visibility_score, contrast, edge_density, haze_strength


# =========================================================
# PREDICT ENDPOINT
# =========================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

        visibility_score, contrast, edge_density,  haze_strength = calculate_visibility_score(image)

        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        label_map = {0: "clear", 1: "hazy"}
        prediction = label_map[predicted.item()]
        confidence_value = round(confidence.item() * 100, 2)

    

        response = {
            "success": True,
            "prediction": prediction,
            "confidence": confidence_value,
            "visibility_score": round(visibility_score, 2),
            "metrics": {
                "contrast": round(contrast, 2),
                "edge_density": round(edge_density * 100, 2),
                "haze_strength": round(haze_strength, 2)

            },
            "timestamp": datetime.utcnow().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/')
def home():
    return jsonify({
        'message': 'CrowdVision Haze Detection API is running!',
        'status': 'active'
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)







































