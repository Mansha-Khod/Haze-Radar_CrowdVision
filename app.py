import os
import gdown
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import cv2
from datetime import datetime

# =====================================================================
# FLASK SETUP
# =====================================================================
app = Flask(__name__)
CORS(app)

# =====================================================================
# MODEL SETUP
# =====================================================================

class CrowdVisionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CrowdVisionModel, self).__init__()
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

# =====================================================================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# =====================================================================

def download_model():
    model_path = os.getenv("MODEL_PATH", "crowd_vision_model.pth")
    file_id = os.getenv("GOOGLE_DRIVE_FILE_ID")
    
    if not os.path.exists(model_path):
        print("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, model_path, quiet=False, use_cookies=False)
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            os.system(f"wget --no-check-certificate '{url}' -O {model_path}")
            if os.path.exists(model_path):
                print("✅ Model downloaded successfully via wget!")
            else:
                print("🚨 Model download failed. Check Google Drive link!")

download_model()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CrowdVisionModel(num_classes=2).to(device)

try:
    checkpoint = torch.load(os.getenv("MODEL_PATH", "crowd_vision_model.pth"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")

# =====================================================================
# IMAGE PREPROCESSING
# =====================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def calculate_visibility_score(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    contrast = gray.std()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    brightness = np.mean(gray)
    visibility_score = min(100, (contrast * 1.5 + edge_density * 300 + (brightness - 128) * 0.2))
    return max(0, visibility_score), contrast, edge_density, brightness

# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Case 1 — user uploaded a file (multipart/form-data)
        if 'image' in request.files:
            image_file = request.files['image']
            image = Image.open(image_file).convert('RGB')

        # Case 2 — frontend sends base64 JSON
        elif request.is_json:
            data = request.get_json(silent=True, force=True)
            if 'image_base64' in data:
                image_data = base64.b64decode(data['image_base64'])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                return jsonify({'error': 'Missing image_base64 field'}), 400
        else:
            return jsonify({'error': 'No image provided'}), 400

        # Convert PIL image → NumPy
        image_array = np.array(image)

        # Calculate visibility
        visibility_score, contrast, edge_density, brightness = calculate_visibility_score(image_array)

        # Preprocess for model
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        label_map = {0: 'clear', 1: 'hazy'}
        prediction = label_map[predicted.item()]
        confidence = round(confidence.item() * 100, 2)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'visibility_score': round(visibility_score, 2),
            'metrics': {
                'contrast': round(contrast, 2),
                'edge_density': round(edge_density * 100, 2),
                'brightness': round(brightness, 2)
            },
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/')
def home():
    return jsonify({
        'message': 'CrowdVision Haze Detection API is running!',
        'endpoints': {'/predict': 'POST (image upload)', '/health': 'GET'},
        'status': 'active'
    })

# =====================================================================
# RUN FLASK APP
# =====================================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)























