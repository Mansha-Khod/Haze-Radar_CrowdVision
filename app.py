"""
CrowdVision Inference API

"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import base64
import numpy as np
import cv2
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for dashboard integration

# ============================================================================
# MODEL DEFINITION (Same as training)
# ============================================================================

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

# ============================================================================
# LOAD MODEL
# ============================================================================

import gdown
import os

def download_model():
    if not os.path.exists('crowd_vision_model.pth'):
        print("📥 Downloading model from Google Drive...")
        # Your Google Drive file ID
        file_id = "1fCuv2AzaLToXcMq9x7ZVCafY2Ks5CN2J"
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, 'crowd_vision_model.pth', quiet=False)
        print("✅ Model downloaded successfully!")

# Download model first
download_model()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CrowdVisionModel(num_classes=2).to(device)

# Load trained weights
try:
    checkpoint = torch.load('crowd_vision_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_visibility_score(image_array):
    """
    Calculate visibility metrics from image
    Returns score from 0 (very hazy) to 100 (very clear)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Calculate contrast (standard deviation)
    contrast = gray.std()
    
    # Calculate edge density (clear images have more edges)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Calculate brightness
    brightness = np.mean(gray)
    
    # Combine metrics into visibility score
    visibility_score = min(100, (contrast * 1.5 + edge_density * 300 + (brightness - 128) * 0.2))
    visibility_score = max(0, visibility_score)
    
    return visibility_score, contrast, edge_density, brightness

def get_health_advice(prediction, confidence, visibility_score):
    """
    Generate health advice based on prediction and visibility
    """
    if prediction == 'clear':
        if confidence > 0.85 and visibility_score > 70:
            return {
                'status': 'excellent',
                'advice': 'Air quality looks excellent! Perfect time for outdoor activities.',
                'recommendations': [
                    '✓ Great for outdoor exercise',
                    '✓ Safe for children to play outside',
                    '✓ Ideal for morning/evening walks'
                ],
                'icon': '☀️',
                'color': '#10b981'  # green
            }
        elif confidence > 0.7:
            return {
                'status': 'good',
                'advice': 'Sky appears mostly clear. Generally safe for outdoor activities.',
                'recommendations': [
                    '✓ Safe for outdoor activities',
                    '✓ Monitor air quality if exercising intensely',
                    '✓ Sensitive individuals should be cautious'
                ],
                'icon': '⛅',
                'color': '#84cc16'  # light green
            }
        else:
            return {
                'status': 'moderate',
                'advice': 'Conditions are uncertain. Check local air quality readings.',
                'recommendations': [
                    '⚠ Check AQI before prolonged outdoor activity',
                    '⚠ Sensitive groups should limit outdoor exposure',
                    '⚠ Consider indoor alternatives if feeling discomfort'
                ],
                'icon': '🌤️',
                'color': '#facc15'  # yellow
            }
    else:  # hazy
        if confidence > 0.85 and visibility_score < 40:
            return {
                'status': 'unhealthy',
                'advice': 'Significant haze detected. Refrain from outdoor activities.',
                'recommendations': [
                    '✗ Avoid outdoor exercise',
                    '✗ Keep windows closed',
                    '✗ Use air purifiers indoors',
                    '✗ Wear N95 masks if going outside is necessary',
                    '✗ Vulnerable groups should stay indoors'
                ],
                'icon': '🌫️',
                'color': '#ef4444'  # red
            }
        elif confidence > 0.7:
            return {
                'status': 'unhealthy_sensitive',
                'advice': 'Haze detected. Limit outdoor activities, especially for sensitive groups.',
                'recommendations': [
                    '⚠ Limit prolonged outdoor activities',
                    '⚠ Sensitive individuals should stay indoors',
                    '⚠ Close windows during peak haze hours',
                    '⚠ Consider wearing masks outdoors'
                ],
                'icon': '😷',
                'color': '#f97316'  # orange
            }
        else:
            return {
                'status': 'moderate',
                'advice': 'Possible haze detected. Monitor conditions and limit outdoor exposure.',
                'recommendations': [
                    '⚠ Check local air quality updates',
                    '⚠ Reduce outdoor activities if feeling discomfort',
                    '⚠ Keep emergency contacts handy'
                ],
                'icon': '🌥️',
                'color': '#facc15'  # yellow
            }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts: image file or base64 encoded image
    Returns: prediction, confidence, visibility metrics, and health advice
    """
    try:
        # Get image from request
        if 'image' in request.files:
            # File upload
            image_file = request.files['image']
            image = Image.open(image_file).convert('RGB')
        elif 'image_base64' in request.json:
            # Base64 encoded image
            image_data = base64.b64decode(request.json['image_base64'])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert PIL to numpy for visibility analysis
        image_array = np.array(image)
        
        # Calculate visibility metrics
        visibility_score, contrast, edge_density, brightness = calculate_visibility_score(image_array)
        
        # Preprocess for model
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted_class = predicted.item()
        
        # Map to labels
        label_map = {0: 'clear', 1: 'hazy'}
        prediction = label_map[predicted_class]
        
        # Get health advice
        health_info = get_health_advice(prediction, confidence, visibility_score)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'visibility_score': round(visibility_score, 2),
            'metrics': {
                'contrast': round(contrast, 2),
                'edge_density': round(edge_density * 100, 2),
                'brightness': round(brightness, 2)
            },
            'health': health_info,
            'timestamp': datetime.utcnow().isoformat(),
            'location': request.json.get('location', 'Unknown') if request.json else 'Unknown'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple images
    Accepts: list of base64 encoded images
    """
    try:
        images_data = request.json.get('images', [])
        
        if not images_data:
            return jsonify({'error': 'No images provided'}), 400
        
        results = []
        
        for idx, img_data in enumerate(images_data):
            try:
                # Decode image
                image_bytes = base64.b64decode(img_data['image_base64'])
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                image_array = np.array(image)
                
                # Calculate metrics
                visibility_score, _, _, _ = calculate_visibility_score(image_array)
                
                # Predict
                image_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                label_map = {0: 'clear', 1: 'hazy'}
                prediction = label_map[predicted.item()]
                
                results.append({
                    'image_id': img_data.get('id', idx),
                    'prediction': prediction,
                    'confidence': round(confidence.item() * 100, 2),
                    'visibility_score': round(visibility_score, 2)
                })
            
            except Exception as e:
                results.append({
                    'image_id': img_data.get('id', idx),
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# RUN APP
# ============================================================================
@app.route('/')
def home():
    return jsonify({
        'message': 'CrowdVision Haze Detection API is running!',
        'endpoints': {
            'health': '/health (GET)',
            'predict': '/predict (POST)',
            'batch_predict': '/batch_predict (POST)'
        },
        'status': 'active'
    })
if __name__ == '__main__':
    
    app.run(debug=False, host='0.0.0.0', port=7860)














