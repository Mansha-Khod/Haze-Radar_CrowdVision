import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)

# Define model
class HazeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # 2 classes: clear, hazy
        )

    def forward(self, x):
        return self.backbone(x)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HazeDetector().to(device)
model.load_state_dict(torch.load("haze_detector_colab.pth", map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = probs.max(1)
        confidence = conf.item()
        label = 'Hazy' if pred.item() == 1 else 'Clear'

    advice = ("Stay indoors! Air is polluted." if label == "Hazy"
              else "Air is clean. Enjoy the outdoors!")

    return jsonify({
        'condition': label,
        'confidence': round(confidence * 100, 2),
        'advice': advice
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'CrowdVision Haze Detector'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)








