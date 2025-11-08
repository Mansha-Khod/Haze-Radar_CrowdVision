import os
import io
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
import gdown

app = Flask(__name__)
CORS(app)
device = torch.device("cpu")

# ===============================================================
# Download Model
# ===============================================================
MODEL_PATH = "crowd_vision_model.pth"
DRIVE_FILE_ID = "1oP7BfyzGLU83KbIzOlmuQM8qBVW6XheT"

if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)  # Force fresh download
    print("🗑️ Removed old model file")

print("📥 Downloading YOUR trained model from Google Drive...")
gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
print(" Model downloaded successfully!")

# ===============================================================
# Model Architecture (WITHOUT Pretrained Weights)
# ===============================================================
class CrowdVisionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CrowdVisionModel, self).__init__()
        # 🚨 NO PRETRAINED WEIGHTS - we'll load YOUR custom weights
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Initialize model
model = CrowdVisionModel(num_classes=2).to(device)
print(" Model architecture created")

# Load YOUR trained weights
print(" Loading YOUR trained haze detection weights...")
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print(" SUCCESS: Your custom haze detection weights loaded!")
    
    # Verify the weights are working
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        test_output = model(dummy_input)
        test_probs = torch.softmax(test_output, dim=1)
    print(f" Verification - Output probabilities: {test_probs}")
    
    # They should NOT be 50/50 if your model trained properly!
    if abs(test_probs[0][0] - test_probs[0][1]) < 0.3:
        print(" WARNING: Model still giving similar probabilities - training might have issues")
    
except Exception as e:
    print(f"❌ FAILED to load your weights: {e}")

model.eval()

# Rest of your code (transforms, visibility, predict endpoint) remains the same...
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
            
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        
        vis_score, contrast, edges, brightness = calculate_visibility_score(image)
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        clear_prob = probs[0][0].item()
        hazy_prob = probs[0][1].item()
        
        print(f" YOUR MODEL - Clear: {clear_prob*100:.2f}% | Hazy: {hazy_prob*100:.2f}%")
        print(f" YOUR MODEL - Raw outputs: {outputs.cpu().numpy()}")
        
        label_map = {0: "clear", 1: "hazy"}
        prediction = label_map[predicted.item()]
        confidence_value = float(confidence.item() * 100)
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence_value, 2),
            "visibility_score": round(vis_score, 2),
            "probabilities": {
                "clear": round(clear_prob * 100, 2),
                "hazy": round(hazy_prob * 100, 2)
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
    return {"status": "CrowdVision HazeRadar - USING YOUR TRAINED WEIGHTS"}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
